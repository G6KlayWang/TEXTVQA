from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image

from src.data.dataset import normalize_raw_row, split_alias
from src.utils.config import ensure_dir, load_yaml
from src.utils.io import write_json, write_jsonl


os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess TextVQA into JSONL plus cached images.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_dataset_from_config(config: dict[str, Any]):
    dataset_dir = Path(config["cache_dir"]) / "hf_dataset"
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:
        raise SystemExit("Install the `datasets` package before preprocessing TextVQA.") from exc

    if dataset_dir.exists():
        return load_from_disk(str(dataset_dir))
    selected_splits = config.get("splits") or ["train", "validation"]
    return {split: load_dataset(config["hf_id"], split=split, cache_dir=config["cache_dir"]) for split in selected_splits}


def get_image(row: dict[str, Any]) -> Image.Image:
    image = row.get("image")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, dict):
        path = image.get("path") or image.get("bytes")
        if path and isinstance(path, str):
            return Image.open(path).convert("RGB")
    for key in ["image_path", "path"]:
        value = row.get(key)
        if isinstance(value, str) and Path(value).exists():
            return Image.open(value).convert("RGB")
    raise ValueError("Could not locate an image field in dataset row.")


def save_image(image: Image.Image, output_dir: Path, image_id: str, image_size: int | None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = SAFE_NAME_RE.sub("_", image_id)[:160]
    path = output_dir / f"{safe_id}.jpg"
    if image_size:
        image = image.copy()
        image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    image.save(path, format="JPEG", quality=95)
    return path


def preprocess_split(dataset, split: str, config: dict[str, Any]) -> dict[str, Any]:
    out_split = split_alias(split)
    processed_dir = ensure_dir(config["processed_dir"])
    output_path = processed_dir / f"{out_split}.jsonl"
    image_dir = processed_dir / "images" / out_split

    max_key = f"max_{out_split}_samples"
    if out_split == "val":
        max_samples = config.get(max_key, config.get("max_validation_samples"))
    else:
        max_samples = config.get(max_key)

    rows = []
    failures = 0
    limit = len(dataset) if max_samples in (None, "null") else min(int(max_samples), len(dataset))
    for index in range(limit):
        raw = dataset[index]
        sample = normalize_raw_row(raw, index, split)
        try:
            image = get_image(raw)
            image_path = save_image(image, image_dir, sample["image_id"], config.get("image_size"))
        except Exception as exc:
            failures += 1
            print(f"Skipping {sample['id']}: {exc}")
            continue
        sample["image_path"] = str(image_path)
        rows.append(sample)

    write_jsonl(rows, output_path)
    return {"split": out_split, "samples": len(rows), "failures": failures, "path": str(output_path)}


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    dataset = load_dataset_from_config(config)
    manifest = []
    for split in config.get("splits", ["train", "validation"]):
        if split not in dataset:
            print(f"Skipping missing split {split}; available splits: {list(dataset.keys())}")
            continue
        manifest.append(preprocess_split(dataset[split], split, config))
    write_json({"splits": manifest}, Path(config["processed_dir"]) / "preprocess_manifest.json")
    print(f"Preprocessed {sum(item['samples'] for item in manifest)} samples.")


if __name__ == "__main__":
    main()
