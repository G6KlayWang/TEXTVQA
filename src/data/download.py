from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.config import ensure_dir, load_yaml
from src.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/cache TextVQA from Hugging Face.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    cache_dir = ensure_dir(config["cache_dir"])
    dataset_dir = cache_dir / "hf_dataset"

    if dataset_dir.exists() and not args.force:
        print(f"Dataset already cached at {dataset_dir}")
        return

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the `datasets` package before downloading TextVQA.") from exc

    dataset = load_dataset(config["hf_id"], cache_dir=str(cache_dir))
    selected_splits = config.get("splits") or list(dataset.keys())
    selected = {split: dataset[split] for split in selected_splits if split in dataset}
    if not selected:
        raise SystemExit(f"No requested splits found. Requested={selected_splits}; available={list(dataset.keys())}")

    from datasets import DatasetDict

    DatasetDict(selected).save_to_disk(str(dataset_dir))
    write_json(
        {
            "hf_id": config["hf_id"],
            "splits": {split: len(ds) for split, ds in selected.items()},
            "path": str(Path(dataset_dir)),
        },
        cache_dir / "download_manifest.json",
    )
    print(f"Cached {config['hf_id']} at {dataset_dir}")


if __name__ == "__main__":
    main()

