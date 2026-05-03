from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def load_prompt_template(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def render_prompt(template: str, sample: dict[str, Any]) -> str:
    return template.format(
        question=sample.get("question", ""),
        ocr_tokens=" ".join(sample.get("ocr_tokens") or []),
        answers=", ".join(sample.get("answers") or []),
    )


def build_messages(sample: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    image = sample.get("image")
    if image is None:
        image = Image.open(sample["image_path"]).convert("RGB")
    return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]


class QwenGenerator:
    def __init__(self, model, processor) -> None:
        self.model = model
        self.processor = processor
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise SystemExit("Install `qwen-vl-utils` before running inference.") from exc
        self.process_vision_info = process_vision_info

    def generate_one(self, sample: dict[str, Any], prompt: str, decoding: dict[str, Any]) -> str:
        import torch

        messages = build_messages(sample, prompt)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = getattr(self.model, "device", None)
        if device is None:
            device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **decoding)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = generated_ids[:, input_len:]
        output = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output.strip()
