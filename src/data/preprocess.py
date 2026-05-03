from __future__ import annotations

import argparse
import io
import multiprocessing as mp
import os
import re
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from src.data.dataset import normalize_raw_row, split_alias
from src.utils.config import ensure_dir, load_yaml
from src.utils.io import write_json, write_jsonl


os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_WORKER_DATASET = None
_WORKER_CONFIG: dict[str, Any] | None = None
_WORKER_SPLIT: str | None = None
_WORKER_OUT_SPLIT: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess TextVQA into JSONL plus cached images.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_split_from_config(config: dict[str, Any], split: str):
    dataset_dir = Path(config["cache_dir"]) / "hf_dataset"
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:
        raise SystemExit("Install the `datasets` package before preprocessing TextVQA.") from exc

    if dataset_dir.exists():
        dataset = load_from_disk(str(dataset_dir))
        return dataset[split]
    return load_dataset(config["hf_id"], split=split, cache_dir=config["cache_dir"])


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
        path = image.get("path")
        if isinstance(path, str):
            return Image.open(path).convert("RGB")
        image_bytes = image.get("bytes")
        if isinstance(image_bytes, bytes):
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    for key in ["image_path", "path"]:
        value = row.get(key)
        if isinstance(value, str) and Path(value).exists():
            return Image.open(value).convert("RGB")
    raise ValueError("Could not locate an image field in dataset row.")


def is_valid_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def save_image(image: Image.Image, output_dir: Path, image_id: str, image_size: int | None, force: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = SAFE_NAME_RE.sub("_", image_id)[:160]
    path = output_dir / f"{safe_id}.jpg"
    if not force and is_valid_image(path):
        return path
    if image_size:
        image = image.copy()
        image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    tmp_path = output_dir / f".{safe_id}.{os.getpid()}.{uuid.uuid4().hex}.tmp.jpg"
    image.save(tmp_path, format="JPEG", quality=95)
    os.replace(tmp_path, path)
    return path


def init_worker(config: dict[str, Any], split: str, out_split: str) -> None:
    global _WORKER_CONFIG, _WORKER_DATASET, _WORKER_OUT_SPLIT, _WORKER_SPLIT
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    _WORKER_CONFIG = config
    _WORKER_SPLIT = split
    _WORKER_OUT_SPLIT = out_split
    _WORKER_DATASET = load_split_from_config(config, split)


def process_index(index: int) -> tuple[dict[str, Any] | None, str | None]:
    if _WORKER_DATASET is None or _WORKER_CONFIG is None or _WORKER_SPLIT is None or _WORKER_OUT_SPLIT is None:
        raise RuntimeError("Worker was not initialized.")
    processed_dir = Path(_WORKER_CONFIG["processed_dir"])
    image_dir = processed_dir / "images" / _WORKER_OUT_SPLIT
    raw = _WORKER_DATASET[index]
    sample = normalize_raw_row(raw, index, _WORKER_SPLIT)
    try:
        image = get_image(raw)
        image_path = save_image(
            image,
            image_dir,
            sample["image_id"],
            _WORKER_CONFIG.get("image_size"),
            bool(_WORKER_CONFIG.get("force", False)),
        )
    except Exception as exc:
        return None, f"Skipping {sample['id']}: {exc}"
    sample["image_path"] = str(image_path)
    sample["_index"] = index
    return sample, None


def process_index_serial(dataset, split: str, out_split: str, config: dict[str, Any], index: int) -> tuple[dict | None, str | None]:
    processed_dir = Path(config["processed_dir"])
    image_dir = processed_dir / "images" / out_split
    raw = dataset[index]
    sample = normalize_raw_row(raw, index, split)
    try:
        image = get_image(raw)
        image_path = save_image(
            image,
            image_dir,
            sample["image_id"],
            config.get("image_size"),
            bool(config.get("force", False)),
        )
    except Exception as exc:
        return None, f"Skipping {sample['id']}: {exc}"
    sample["image_path"] = str(image_path)
    sample["_index"] = index
    return sample, None


def write_jsonl_atomic(rows: list[dict[str, Any]], output_path: Path) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + f".{os.getpid()}.tmp")
    write_jsonl(rows, tmp_path)
    os.replace(tmp_path, output_path)


def progress_bar(iterable, *, total: int, desc: str):
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit="sample",
        dynamic_ncols=True,
        mininterval=0.5,
        file=sys.stdout,
        leave=True,
    )


def preprocess_split(dataset, split: str, config: dict[str, Any], num_workers: int) -> dict[str, Any]:
    out_split = split_alias(split)
    processed_dir = ensure_dir(config["processed_dir"])
    output_path = processed_dir / f"{out_split}.jsonl"

    max_key = f"max_{out_split}_samples"
    if out_split == "val":
        max_samples = config.get(max_key, config.get("max_validation_samples"))
    else:
        max_samples = config.get(max_key)

    rows = []
    failures = 0
    limit = len(dataset) if max_samples in (None, "null") else min(int(max_samples), len(dataset))
    indices = range(limit)
    print(
        f"Preprocessing split={out_split} samples={limit} workers={num_workers} "
        f"image_size={config.get('image_size')} output={output_path}",
        flush=True,
    )
    if num_workers <= 1:
        with progress_bar(indices, total=limit, desc=f"preprocess {out_split}") as pbar:
            for index in pbar:
                sample, error = process_index_serial(dataset, split, out_split, config, index)
                if error:
                    failures += 1
                    tqdm.write(error, file=sys.stdout)
                elif sample is not None:
                    rows.append(sample)
                pbar.set_postfix(saved=len(rows), failed=failures, refresh=False)
    else:
        ctx_name = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context(ctx_name),
            initializer=init_worker,
            initargs=(config, split, out_split),
        ) as executor:
            with progress_bar(
                executor.map(process_index, indices, chunksize=16),
                total=limit,
                desc=f"preprocess {out_split} ({num_workers} workers)",
            ) as pbar:
                for sample, error in pbar:
                    if error:
                        failures += 1
                        tqdm.write(error, file=sys.stdout)
                    elif sample is not None:
                        rows.append(sample)
                    pbar.set_postfix(saved=len(rows), failed=failures, refresh=False)

    rows.sort(key=lambda row: int(row.get("_index", 0)))
    for row in rows:
        row.pop("_index", None)
    if len(rows) != limit:
        failures = max(failures, limit - len(rows))
    write_jsonl_atomic(rows, output_path)
    print(
        f"Finished split={out_split}: saved={len(rows)} failed={failures} jsonl={output_path}",
        flush=True,
    )
    return {
        "split": out_split,
        "samples": len(rows),
        "failures": failures,
        "path": str(output_path),
        "num_workers": num_workers,
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    config["force"] = bool(args.force)
    configured_workers = config.get("preprocess_num_workers")
    num_workers = args.num_workers if args.num_workers is not None else configured_workers
    if num_workers in (None, "auto"):
        num_workers = min(os.cpu_count() or 1, 8)
    num_workers = max(int(num_workers), 1)
    dataset = load_dataset_from_config(config)
    manifest = []
    for split in config.get("splits", ["train", "validation"]):
        if split not in dataset:
            print(f"Skipping missing split {split}; available splits: {list(dataset.keys())}")
            continue
        manifest.append(preprocess_split(dataset[split], split, config, num_workers))
    write_json({"splits": manifest}, Path(config["processed_dir"]) / "preprocess_manifest.json")
    print(f"Preprocessed {sum(item['samples'] for item in manifest)} samples.")


if __name__ == "__main__":
    main()
