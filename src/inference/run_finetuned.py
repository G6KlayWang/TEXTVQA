from __future__ import annotations

import argparse

from src.inference.run_zeroshot import clean_decoding
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process = current_process()
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise SystemExit("Install `peft` before loading a LoRA adapter.") from exc
    from tqdm import tqdm

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
    rows = []
    limit = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    indices = list(shard_indices(limit, process.rank, process.world_size))
    for idx in tqdm(indices, desc=f"finetuned rank {process.rank}", disable=not process.is_main):
        sample = dataset[idx]
        prompt = render_prompt(template, sample)
        prediction = generator.generate_one(sample, prompt, decoding)
        rows.append(
            {k: sample[k] for k in sample if k != "image"}
            | {"prediction": prediction, "prompt": prompt, "_index": idx}
        )
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
