from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.data.dataset import ProcessedTextVQADataset, processed_path
from src.inference.generate import load_prompt_template, render_prompt
from src.inference.run_zeroshot import IndexedSubset, clean_decoding, collate_samples, progress
from src.models.load_blip2 import load_blip2_model_and_processor
from src.utils.config import load_yaml
from src.utils.distributed import cleanup_parts, current_process, merge_jsonl_parts, part_path, shard_indices
from src.utils.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BLIP-2 inference on TextVQA.")
    parser.add_argument("--model_config", default="configs/model_blip2_opt27b.yaml")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--eval_config", default="configs/eval_blip2.yaml")
    parser.add_argument("--data_config", default="configs/data.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", default="artifacts/predictions/blip2_zero_shot_val.jsonl")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--show_all_progress", action="store_true")
    return parser.parse_args()


def model_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def generate_one(model, processor, sample: dict, prompt: str, decoding: dict) -> str:
    import torch

    inputs = processor(images=sample["image"], text=prompt, return_tensors="pt").to(model_device(model))
    with torch.no_grad():
        generated = model.generate(**inputs, **decoding)
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def main() -> None:
    args = parse_args()
    process = current_process()
    model_config = load_yaml(args.model_config)
    eval_config = load_yaml(args.eval_config)
    data_config = load_yaml(args.data_config)
    dataset = ProcessedTextVQADataset(processed_path(data_config["processed_dir"], args.split))
    model, processor = load_blip2_model_and_processor(model_config, for_training=False)
    if args.adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit("Install `peft` before loading a LoRA adapter.") from exc
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    template = load_prompt_template(eval_config.get("prompt_template", "src/prompts/blip2_default.txt"))
    decoding = clean_decoding(eval_config.get("decoding", {}))
    batch_size = int(eval_config.get("batch_size", 1) or 1)
    configured_workers = eval_config.get("dataloader_num_workers")
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = int(os.environ.get("EVAL_NUM_WORKERS", configured_workers or 2))
    num_workers = max(int(num_workers), 0)
    show_all_progress = args.show_all_progress or os.environ.get("SHOW_ALL_PROGRESS", "1") != "0"

    limit = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    indices = list(shard_indices(limit, process.rank, process.world_size))
    label = "blip2-lora" if args.adapter_path else "blip2-zero"
    if process.is_main:
        print(
            f"BLIP-2 eval: total={limit} world_size={process.world_size} "
            f"batch_size={batch_size} cpu_workers_per_rank={num_workers} adapter={bool(args.adapter_path)}",
            flush=True,
        )

    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("Install PyTorch before running inference.") from exc

    loader = DataLoader(
        IndexedSubset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_samples,
        persistent_workers=num_workers > 0,
    )
    rows = []
    with progress(loader, total=len(indices), process=process, show_all=show_all_progress, label=label) as pbar:
        for batch in pbar:
            for sample in batch:
                idx = sample["_index"]
                prompt = render_prompt(template, sample)
                prediction = generate_one(model, processor, sample, prompt, decoding)
                rows.append(
                    {k: sample[k] for k in sample if k not in {"image", "_index"}}
                    | {"prediction": prediction, "prompt": prompt, "_index": idx}
                )
            pbar.update(len(batch) - 1)
            pbar.set_postfix(written=len(rows), refresh=False)

    output_path = Path(args.output)
    if process.distributed:
        write_jsonl(rows, part_path(output_path, process.rank))
        process.wait()
        if process.is_main:
            merge_jsonl_parts(output_path, process.world_size)
            cleanup_parts(output_path, range(process.world_size))
            print(f"Wrote {limit} merged predictions to {args.output}")
        process.wait()
    else:
        for row in rows:
            row.pop("_index", None)
        write_jsonl(rows, output_path)
        print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()

