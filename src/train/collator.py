from __future__ import annotations

from typing import Any


def build_training_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {
                    "type": "text",
                    "text": f"Question: {sample['question']}\nAnswer with the shortest correct text from the image.",
                },
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample.get("target_answer", "")}]},
    ]


class QwenVLCollator:
    def __init__(self, processor, max_length: int | None = None) -> None:
        self.processor = processor
        self.max_length = max_length
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise SystemExit("Install `qwen-vl-utils` before training/inference.") from exc
        self.process_vision_info = process_vision_info

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        texts = []
        image_inputs = []
        video_inputs = []
        for sample in features:
            messages = build_training_messages(sample)
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
            images, videos = self.process_vision_info(messages)
            image_inputs.extend(images or [])
            if videos:
                video_inputs.extend(videos)

        inputs = self.processor(
            text=texts,
            images=image_inputs or None,
            videos=video_inputs or None,
            padding=True,
            truncation=bool(self.max_length),
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        inputs["labels"] = labels
        return inputs

