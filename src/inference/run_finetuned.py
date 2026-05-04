from __future__ import annotations

import argparse
import os

from src.inference.run_zeroshot import IndexedSubset, clean_decoding, collate_samples, progress
from src.data.dataset import ProcessedTextVQADataset, processed_path
from src.inference.generate import QwenGenerator, load_prompt_template, render_prompt
from src.models.load_qwen import load_qwen_model_and_processor
from src.utils.config import load_yaml
from src.utils.distributed import cleanup_parts, current_process, merge_jsonl_parts, part_path, shard_indices
from src.utils.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TextVQA inference with a LoRA adapter.")
    parser.add_argument("--model_config", default="configs/model_qwen25vl.yaml")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--eval_config", default="configs/eval.yaml")
    parser.add_argument("--data_config", default="configs/data.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", default="artifacts/predictions/finetuned_val.jsonl")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--show_all_progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process = current_process()
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise SystemExit("Install `peft` before loading a LoRA adapter.") from exc

    model_config = load_yaml(args.model_config)
    eval_config = load_yaml(args.eval_config)
    data_config = load_yaml(args.data_config)
    dataset = ProcessedTextVQADataset(processed_path(data_config["processed_dir"], args.split))
    model, processor = load_qwen_model_and_processor(model_config, for_training=False)
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    generator = QwenGenerator(model, processor)
    template = load_prompt_template(eval_config.get("prompt_template", "src/prompts/answer_only.txt"))
    decoding = clean_decoding(eval_config.get("decoding", {}))
    batch_size = int(eval_config.get("batch_size", 1) or 1)
    configured_workers = eval_config.get("dataloader_num_workers")
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = int(os.environ.get("EVAL_NUM_WORKERS", configured_workers or 2))
    num_workers = max(int(num_workers), 0)
    show_all_progress = args.show_all_progress or os.environ.get("SHOW_ALL_PROGRESS", "1") != "0"

    rows = []
    limit = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    indices = list(shard_indices(limit, process.rank, process.world_size))
    if process.is_main:
        print(
            f"Fine-tuned eval: total={limit} world_size={process.world_size} "
            f"batch_size={batch_size} cpu_workers_per_rank={num_workers}",
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
    with progress(loader, total=len(indices), process=process, show_all=show_all_progress, label="finetuned") as pbar:
        for batch in pbar:
            for sample in batch:
                idx = sample["_index"]
                prompt = render_prompt(template, sample)
                prediction = generator.generate_one(sample, prompt, decoding)
                rows.append(
                    {k: sample[k] for k in sample if k not in {"image", "_index"}}
                    | {"prediction": prediction, "prompt": prompt, "_index": idx}
                )
            pbar.update(len(batch) - 1)
            pbar.set_postfix(written=len(rows), refresh=False)

    if process.distributed:
        write_jsonl(rows, part_path(args.output, process.rank))
        process.wait()
        if process.is_main:
            merge_jsonl_parts(args.output, process.world_size)
            cleanup_parts(args.output, range(process.world_size))
            print(f"Wrote {limit} merged predictions to {args.output}")
        process.wait()
    else:
        for row in rows:
            row.pop("_index", None)
        write_jsonl(rows, args.output)
        print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
