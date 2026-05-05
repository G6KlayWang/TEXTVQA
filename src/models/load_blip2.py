from __future__ import annotations

import os
from typing import Any

from src.models.load_qwen import torch_dtype


os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def local_files_only(config: dict[str, Any]) -> bool:
    if "local_files_only" in config:
        return bool(config["local_files_only"])
    return os.environ.get("TRANSFORMERS_OFFLINE") == "1" or os.environ.get("HF_HUB_OFFLINE") == "1"


def load_blip2_model_and_processor(config: dict[str, Any], for_training: bool = False):
    try:
        from transformers import AutoProcessor, Blip2ForConditionalGeneration
    except ImportError as exc:
        raise SystemExit("Install `transformers` before loading BLIP-2.") from exc

    model_id = config["model_id"]
    offline = local_files_only(config)
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=offline)

    device_map = config.get("device_map", "auto")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed and for_training and not config.get("load_in_4bit"):
        device_map = None
    elif distributed and device_map == "auto":
        device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype(config.get("dtype")),
        "local_files_only": offline,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    if config.get("load_in_4bit"):
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise SystemExit("Install `bitsandbytes` support before using load_in_4bit.") from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype(config.get("dtype", "bfloat16")),
        )

    model = Blip2ForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    if for_training:
        model.config.use_cache = False
    return model, processor

