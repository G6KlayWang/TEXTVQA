from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from src.utils.io import read_jsonl


def split_alias(split: str) -> str:
    return {"validation": "val", "valid": "val"}.get(split, split)


def pick_first(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def coerce_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                answer = pick_first(item, ["answer", "text", "label"])
                if answer is not None:
                    out.append(str(answer))
            else:
                out.append(str(item))
        return out
    return [str(value)]


def coerce_ocr_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        tokens: list[str] = []
        for item in value:
            if isinstance(item, str):
                tokens.append(item)
            elif isinstance(item, dict):
                token = pick_first(item, ["text", "token", "word"])
                if token is not None:
                    tokens.append(str(token))
        return tokens
    return []


def majority_answer(answers: list[str]) -> str:
    if not answers:
        return ""
    counts = Counter(a.strip() for a in answers if a and a.strip())
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def normalize_raw_row(row: dict[str, Any], index: int, split: str) -> dict[str, Any]:
    question_id = pick_first(row, ["question_id", "questionId", "qid", "id"], f"{split}-{index}")
    image_id = pick_first(row, ["image_id", "imageId"], question_id)
    question = str(pick_first(row, ["question", "query"], "")).strip()
    answers = coerce_answers(pick_first(row, ["answers", "answer", "labels"], []))
    ocr_tokens = coerce_ocr_tokens(pick_first(row, ["ocr_tokens", "ocr", "ocr_info"], []))
    return {
        "id": str(question_id),
        "question_id": str(question_id),
        "image_id": str(image_id),
        "split": split_alias(split),
        "question": question,
        "answers": answers,
        "target_answer": majority_answer(answers),
        "ocr_tokens": ocr_tokens,
    }


class ProcessedTextVQADataset:
    def __init__(self, jsonl_path: str | Path) -> None:
        self.path = Path(jsonl_path)
        self.rows = list(read_jsonl(self.path))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        image_path = Path(row["image_path"])
        row["image"] = Image.open(image_path).convert("RGB")
        return row


def processed_path(processed_dir: str | Path, split: str) -> Path:
    return Path(processed_dir) / f"{split_alias(split)}.jsonl"

