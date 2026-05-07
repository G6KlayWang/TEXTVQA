from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import read_jsonl, write_json


def parse_prediction_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction inputs must use tag=path format")
    tag, path = value.split("=", 1)
    tag = tag.strip()
    path = path.strip()
    if not tag or not path:
        raise argparse.ArgumentTypeError("prediction inputs must use non-empty tag=path values")
    return tag, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export TextVQA validation samples with images and all model predictions."
    )
    parser.add_argument(
        "--prediction",
        action="append",
        type=parse_prediction_arg,
        required=True,
        help="Prediction file in tag=path format. Repeat for multiple models.",
    )
    parser.add_argument("--reference_tag", default=None, help="Tag to sample from. Defaults to the first prediction.")
    parser.add_argument("--output_dir", default="artifacts/sample_outputs")
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--first", action="store_true", help="Take the first N rows instead of a random sample.")
    return parser.parse_args()


def load_prediction_file(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing prediction file: {path}")
    rows = list(read_jsonl(path))
    if not rows:
        raise SystemExit(f"No rows found in {path}")
    return rows


def row_key(row: dict) -> str:
    return str(row.get("question_id") or row.get("id") or "")


def copied_image_name(row: dict, index: int, source_path: Path) -> str:
    question_id = str(row.get("question_id") or row.get("id") or index)
    return f"{index:02d}_{question_id}{source_path.suffix.lower()}"


def prediction_payload(row: dict | None) -> dict:
    if row is None:
        return {"prediction": None, "available": False}
    payload = {"prediction": row.get("prediction", ""), "available": True}
    if "raw_prediction" in row:
        payload["raw_prediction"] = row.get("raw_prediction", "")
    return payload


def main() -> None:
    args = parse_args()
    prediction_inputs: list[tuple[str, Path]] = args.prediction
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    json_dir = output_dir / "json"
    manifest_path = output_dir / "samples.json"

    prediction_rows = {tag: load_prediction_file(path) for tag, path in prediction_inputs}
    prediction_maps = {
        tag: {row_key(row): row for row in rows if row_key(row)}
        for tag, rows in prediction_rows.items()
    }

    reference_tag = args.reference_tag or prediction_inputs[0][0]
    if reference_tag not in prediction_rows:
        raise SystemExit(f"Unknown --reference_tag {reference_tag!r}")

    reference_rows = prediction_rows[reference_tag]
    if args.num_samples <= 0:
        raise SystemExit("--num_samples must be positive")

    sample_count = min(args.num_samples, len(reference_rows))
    if args.first:
        selected = reference_rows[:sample_count]
    else:
        rng = random.Random(args.seed)
        selected = rng.sample(reference_rows, sample_count)

    image_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for index, reference_row in enumerate(selected, start=1):
        key = row_key(reference_row)
        original_image_path = Path(reference_row.get("image_path", ""))
        source_image = original_image_path
        if not source_image.is_absolute():
            source_image = REPO_ROOT / source_image
        if not source_image.exists():
            raise SystemExit(f"Missing image for question {key}: {original_image_path}")

        image_name = copied_image_name(reference_row, index, source_image)
        shutil.copy2(source_image, image_dir / image_name)

        predictions = {
            tag: prediction_payload(prediction_maps[tag].get(key))
            for tag, _ in prediction_inputs
        }
        sample = {
            "id": reference_row.get("id"),
            "question_id": reference_row.get("question_id"),
            "image_id": reference_row.get("image_id"),
            "question": reference_row.get("question", ""),
            "answer": reference_row.get("target_answer", ""),
            "answers": reference_row.get("answers") or [],
            "source_image_path": str(original_image_path),
            "image_path": str(Path("images") / image_name),
            "predictions": predictions,
        }

        sample_json_name = f"{index:02d}_{key}.json"
        write_json(sample, json_dir / sample_json_name)
        samples.append(sample | {"json_path": str(Path("json") / sample_json_name)})

    write_json(
        {
            "num_samples": len(samples),
            "reference_tag": reference_tag,
            "prediction_files": {tag: str(path) for tag, path in prediction_inputs},
            "samples": samples,
        },
        manifest_path,
    )
    print(f"Wrote {len(samples)} samples to {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Per-sample JSON: {json_dir}")
    print(f"Images: {image_dir}")


if __name__ == "__main__":
    main()
