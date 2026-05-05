from __future__ import annotations

from typing import Any

from src.inference.generate import load_prompt_template, render_prompt


class Blip2Collator:
    def __init__(self, processor, prompt_template: str = "src/prompts/blip2_default.txt", max_length: int | None = None):
        self.processor = processor
        self.prompt_template = load_prompt_template(prompt_template)
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        images = [sample["image"] for sample in features]
        prompts = [render_prompt(self.prompt_template, sample) for sample in features]
        answers = [sample.get("target_answer", "") for sample in features]

        inputs = self.processor(
            images=images,
            text=prompts,
            padding=True,
            truncation=bool(self.max_length),
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = self.processor.tokenizer(
            answers,
            padding=True,
            truncation=bool(self.max_length),
            max_length=self.max_length,
            return_tensors="pt",
        ).input_ids
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        inputs["labels"] = labels
        return inputs

