from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.data.dataset import ProcessedTextVQADataset, processed_path
from src.models.load_qwen import load_qwen_model_and_processor
from src.models.lora_setup import apply_lora
from src.train.collator import QwenVLCollator
from src.utils.config import ensure_dir, load_yaml
from src.utils.io import write_json
from src.utils.repro import config_snapshot, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL with LoRA on TextVQA.")
    parser.add_argument("--model_config", default="configs/model_qwen25vl.yaml")
    parser.add_argument("--lora_config", default="configs/lora.yaml")
    parser.add_argument("--train_config", default="configs/train.yaml")
    parser.add_argument("--data_config", default="configs/data.yaml")
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def split_train_eval(dataset: ProcessedTextVQADataset, holdout: int):
    if holdout <= 0 or len(dataset) <= holdout:
        return dataset, None
    try:
        from torch.utils.data import Subset
    except ImportError as exc:
        raise SystemExit("Install PyTorch before training.") from exc
    train_indices = list(range(0, len(dataset) - holdout))
    eval_indices = list(range(len(dataset) - holdout, len(dataset)))
    return Subset(dataset, train_indices), Subset(dataset, eval_indices)


def main() -> None:
    args = parse_args()
    model_config = load_yaml(args.model_config)
    lora_config = load_yaml(args.lora_config)
    train_config = load_yaml(args.train_config)
    data_config = load_yaml(args.data_config)
    output_dir = ensure_dir(args.output_dir or train_config["output_dir"])

    seed_everything(int(train_config.get("seed", 42)))

    train_jsonl = processed_path(data_config["processed_dir"], "train")
    if not train_jsonl.exists():
        raise SystemExit(f"Missing {train_jsonl}. Run `bash scripts/02_preprocess.sh` first.")

    dataset = ProcessedTextVQADataset(train_jsonl)
    train_dataset, eval_dataset = split_train_eval(
        dataset, int(train_config.get("validation_holdout_from_train") or 0)
    )

    model, processor = load_qwen_model_and_processor(model_config, for_training=True)
    model = apply_lora(model, lora_config)
    if train_config.get("gradient_checkpointing", False) and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    try:
        from transformers import Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit("Install `transformers` before training.") from exc

    report_to = train_config.get("report_to") or []
    import inspect

    training_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(train_config["num_train_epochs"]),
        "per_device_train_batch_size": int(train_config["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(train_config["gradient_accumulation_steps"]),
        "learning_rate": float(train_config["learning_rate"]),
        "lr_scheduler_type": train_config.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": float(train_config.get("warmup_ratio", 0.0)),
        "weight_decay": float(train_config.get("weight_decay", 0.0)),
        "bf16": bool(train_config.get("bf16", False)),
        "gradient_checkpointing": bool(train_config.get("gradient_checkpointing", False)),
        "logging_steps": int(train_config.get("logging_steps", 25)),
        "save_steps": int(train_config.get("save_steps", 500)),
        "save_total_limit": int(train_config.get("save_total_limit", 3)),
        "report_to": report_to,
        "remove_unused_columns": False,
        "seed": int(train_config.get("seed", 42)),
    }
    if eval_dataset is not None:
        training_kwargs["eval_steps"] = int(train_config.get("eval_steps", 500))
    strategy_key = "evaluation_strategy"
    if "eval_strategy" in inspect.signature(TrainingArguments).parameters:
        strategy_key = "eval_strategy"
    training_kwargs[strategy_key] = "steps" if eval_dataset is not None else "no"
    training_args = TrainingArguments(**training_kwargs)

    collator = QwenVLCollator(processor, max_length=train_config.get("max_seq_length"))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()

    best_dir = output_dir / "best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(best_dir)
    processor.save_pretrained(best_dir)
    write_json(
        config_snapshot(model=model_config, lora=lora_config, train=train_config, data=data_config),
        output_dir / "training_manifest.json",
    )
    print(f"Saved LoRA adapter and processor to {best_dir}")


if __name__ == "__main__":
    main()
