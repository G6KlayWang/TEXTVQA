from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from src.data.dataset import ProcessedTextVQADataset, processed_path
from src.inference.generate import QwenGenerator, load_prompt_template, render_prompt
from src.models.load_qwen import load_qwen_model_and_processor
from src.utils.config import load_yaml
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
    for idx in tqdm(range(limit), desc="zero-shot"):
        sample = dataset[idx]
        prompt = render_prompt(template, sample)
        prediction = generator.generate_one(sample, prompt, decoding)
        rows.append({k: sample[k] for k in sample if k != "image"} | {"prediction": prediction, "prompt": prompt})

    write_jsonl(rows, Path(args.output))
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()

