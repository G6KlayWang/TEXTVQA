from __future__ import annotations

import argparse
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
    return parser.parse_args()


def clean_decoding(decoding: dict) -> dict:
    decoding = dict(decoding or {})
    if not decoding.get("do_sample", False):
        decoding.pop("temperature", None)
    return decoding


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

    rows = []
    limit = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    indices = list(shard_indices(limit, process.rank, process.world_size))
    for idx in tqdm(indices, desc=f"zero-shot rank {process.rank}", disable=not process.is_main):
        sample = dataset[idx]
        prompt = render_prompt(template, sample)
        prediction = generator.generate_one(sample, prompt, decoding)
        rows.append(
            {k: sample[k] for k in sample if k != "image"}
            | {"prediction": prediction, "prompt": prompt, "_index": idx}
        )

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
