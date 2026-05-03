from __future__ import annotations

import os
from typing import Any


def freeze_vision_tower(model) -> None:
    vision_markers = ("visual", "vision", "image_tower", "vision_model")
    for name, param in model.named_parameters():
        if any(marker in name.lower() for marker in vision_markers):
            param.requires_grad = False


def apply_lora(model, config: dict[str, Any]):
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise SystemExit("Install `peft` before LoRA fine-tuning.") from exc

    if config.get("freeze_vision", True):
        freeze_vision_tower(model)

    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=int(config["r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config.get("lora_dropout", 0.0)),
        bias=config.get("bias", "none"),
        task_type=config.get("task_type", "CAUSAL_LM"),
        target_modules=list(config["target_modules"]),
        modules_to_save=list(config.get("modules_to_save") or []),
    )
    model = get_peft_model(model, lora_config)
    if int(os.environ.get("RANK", "0")) == 0:
        model.print_trainable_parameters()
    return model
