from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tqdm import tqdm

from src.data.dataset import ProcessedTextVQADataset, processed_path
from src.inference.generate import QwenGenerator, load_prompt_template, render_prompt
from src.models.load_qwen import load_qwen_model_and_processor
from src.utils.config import load_yaml
from src.utils.distributed import cleanup_parts, current_process, merge_jsonl_parts, part_path, shard_indices
from src.utils.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-shot Qwen2.5-VL inference on TextVQA.")
    parser.add_argument("--model_config", default="configs/model_qwen25vl.yaml")
    parser.add_argument("--eval_config", default="configs/eval.yaml")
    parser.add_argument("--data_config", default="configs/data.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", default="artifacts/predictions/zero_shot_val.jsonl")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--show_all_progress", action="store_true")
    return parser.parse_args()


def clean_decoding(decoding: dict) -> dict:
    decoding = dict(decoding or {})
    if not decoding.get("do_sample", False):
        decoding.pop("temperature", None)
    return decoding


class IndexedSubset:
    def __init__(self, dataset: ProcessedTextVQADataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        sample = self.dataset[index]
        sample["_index"] = index
        return sample


def collate_samples(batch: list[dict]) -> list[dict]:
    return batch


def progress(iterable, *, total: int, process, show_all: bool, label: str = "zero-shot"):
    return tqdm(
        iterable,
        total=total,
        desc=f"{label} rank {process.rank}/{process.world_size}",
        unit="sample",
        dynamic_ncols=True,
        position=process.rank if show_all else 0,
        file=sys.stdout,
        leave=True,
        disable=(not show_all and not process.is_main),
    )


def main() -> None:
    args = parse_args()
    process = current_process()
    model_config = load_yaml(args.model_config)
    eval_config = load_yaml(args.eval_config)
    data_config = load_yaml(args.data_config)
    dataset = ProcessedTextVQADataset(processed_path(data_config["processed_dir"], args.split))
    model, processor = load_qwen_model_and_processor(model_config, for_training=False)
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
            f"Zero-shot eval: total={limit} world_size={process.world_size} "
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
    with progress(loader, total=len(indices), process=process, show_all=show_all_progress, label="zero-shot") as pbar:
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
