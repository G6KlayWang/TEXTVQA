from __future__ import annotations

import argparse
import inspect
import shutil

from src.data.dataset import ProcessedTextVQADataset, processed_path
from src.models.load_blip2 import load_blip2_model_and_processor
from src.models.lora_setup import apply_lora
from src.train.collator_blip2 import Blip2Collator
from src.train.train_lora import compute_warmup_steps, disable_cache, split_train_eval
from src.utils.config import ensure_dir, load_yaml
from src.utils.distributed import current_process
from src.utils.io import write_json
from src.utils.repro import config_snapshot, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BLIP-2 with LoRA on TextVQA.")
    parser.add_argument("--model_config", default="configs/model_blip2_opt27b.yaml")
    parser.add_argument("--lora_config", default="configs/lora_blip2.yaml")
    parser.add_argument("--train_config", default="configs/train.yaml")
    parser.add_argument("--data_config", default="configs/data.yaml")
    parser.add_argument("--eval_config", default="configs/eval_blip2.yaml")
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process = current_process()
    model_config = load_yaml(args.model_config)
    lora_config = load_yaml(args.lora_config)
    train_config = load_yaml(args.train_config)
    data_config = load_yaml(args.data_config)
    eval_config = load_yaml(args.eval_config)
    output_dir = ensure_dir(args.output_dir or "artifacts/checkpoints/lora-blip2-opt27b")

    seed_everything(int(train_config.get("seed", 42)))
    train_jsonl = processed_path(data_config["processed_dir"], "train")
    if not train_jsonl.exists():
        raise SystemExit(f"Missing {train_jsonl}. Run `bash scripts/02_preprocess.sh` first.")

    dataset = ProcessedTextVQADataset(train_jsonl)
    train_dataset, eval_dataset = split_train_eval(
        dataset, int(train_config.get("validation_holdout_from_train") or 0)
    )

    model, processor = load_blip2_model_and_processor(model_config, for_training=True)
    model = apply_lora(model, lora_config)
    disable_cache(model)
    if train_config.get("gradient_checkpointing", False) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    try:
        from transformers import Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit("Install `transformers` before training.") from exc

    training_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(train_config["num_train_epochs"]),
        "per_device_train_batch_size": int(train_config["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(train_config["gradient_accumulation_steps"]),
        "learning_rate": float(train_config["learning_rate"]),
        "lr_scheduler_type": train_config.get("lr_scheduler_type", "cosine"),
        "warmup_steps": compute_warmup_steps(train_config, train_dataset, process),
        "weight_decay": float(train_config.get("weight_decay", 0.0)),
        "bf16": bool(train_config.get("bf16", False)),
        "gradient_checkpointing": bool(train_config.get("gradient_checkpointing", False)),
        "logging_steps": int(train_config.get("logging_steps", 25)),
        "save_steps": int(train_config.get("save_steps", 500)),
        "save_total_limit": int(train_config.get("save_total_limit", 3)),
        "report_to": train_config.get("report_to") or [],
        "remove_unused_columns": False,
        "seed": int(train_config.get("seed", 42)),
        "dataloader_num_workers": int(train_config.get("dataloader_num_workers", 4)),
        "dataloader_pin_memory": bool(train_config.get("dataloader_pin_memory", True)),
        "ddp_find_unused_parameters": bool(train_config.get("ddp_find_unused_parameters", False)),
    }
    if eval_dataset is not None:
        training_kwargs["eval_steps"] = int(train_config.get("eval_steps", 500))
    strategy_key = "evaluation_strategy"
    if "eval_strategy" in inspect.signature(TrainingArguments).parameters:
        strategy_key = "eval_strategy"
    training_kwargs[strategy_key] = "steps" if eval_dataset is not None else "no"
    training_args = TrainingArguments(**training_kwargs)

    collator = Blip2Collator(
        processor,
        prompt_template=eval_config.get("prompt_template", "src/prompts/blip2_default.txt"),
        max_length=train_config.get("max_seq_length"),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()

    process.wait()
    if process.is_main:
        best_dir = output_dir / "best"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        best_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(best_dir)
        processor.save_pretrained(best_dir)
        write_json(
            config_snapshot(
                model=model_config,
                lora=lora_config,
                train=train_config,
                data=data_config,
                eval=eval_config,
            ),
            output_dir / "training_manifest.json",
        )
        print(f"Saved BLIP-2 LoRA adapter and processor to {best_dir}")
    process.wait()


if __name__ == "__main__":
    main()

