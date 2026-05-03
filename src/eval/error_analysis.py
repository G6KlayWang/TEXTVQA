from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.eval.metrics import exact_match, normalize_answer, token_f1
from src.utils.io import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a heuristic TextVQA error taxonomy.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", default="artifacts/metrics/error_taxonomy.json")
    return parser.parse_args()


def classify_error(row: dict) -> str:
    prediction = row.get("prediction", "")
    answers = row.get("answers") or []
    ocr_tokens = [normalize_answer(token) for token in row.get("ocr_tokens") or []]
    pred_norm = normalize_answer(prediction)
    answer_norms = [normalize_answer(answer) for answer in answers]
    if exact_match(prediction, answers):
        return "correct"
    if not pred_norm:
        return "empty_answer"
    if any(answer and answer in " ".join(ocr_tokens) for answer in answer_norms):
        return "missed_visible_ocr"
    if pred_norm in " ".join(ocr_tokens):
        return "wrong_text_selected"
    if token_f1(prediction, answers) > 0:
        return "partial_match"
    if len(pred_norm.split()) > 6:
        return "verbose_or_hallucinated"
    return "other"


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(args.predictions))
    counts = Counter(classify_error(row) for row in rows)
    examples = {}
    for row in rows:
        category = classify_error(row)
        examples.setdefault(category, [])
        if len(examples[category]) < 5:
            examples[category].append(
                {
                    "id": row.get("id"),
                    "question": row.get("question"),
                    "answers": row.get("answers"),
                    "prediction": row.get("prediction"),
                    "image_path": row.get("image_path"),
                }
            )
    write_json({"counts": dict(counts), "examples": examples}, Path(args.out))
    print(f"Wrote error taxonomy to {args.out}")


if __name__ == "__main__":
    main()

