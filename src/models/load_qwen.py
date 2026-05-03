from __future__ import annotations

import os
from typing import Any

from src.utils.config import parse_pixel_expr


def torch_dtype(name: str | None):
    import torch

    if name in (None, "auto"):
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(str(name).lower(), "auto")


def processor_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for key in ["min_pixels", "max_pixels"]:
        if key in config:
            kwargs[key] = parse_pixel_expr(config[key])
    return kwargs


def load_qwen_model_and_processor(config: dict[str, Any], for_training: bool = False):
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise SystemExit("Install `transformers>=4.49` before loading Qwen2.5-VL.") from exc

    model_id = config["model_id"]
    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs(config))

    device_map = config.get("device_map", "auto")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed and for_training and not config.get("load_in_4bit"):
        device_map = None
    elif distributed and device_map == "auto":
        device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype(config.get("dtype")),
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if config.get("attn_implementation"):
        model_kwargs["attn_implementation"] = config["attn_implementation"]

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

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    except Exception:
        if model_kwargs.get("attn_implementation") == "flash_attention_2":
            model_kwargs["attn_implementation"] = "sdpa"
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        else:
            raise

    if for_training:
        model.config.use_cache = False
    return model, processor
