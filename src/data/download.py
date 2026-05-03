from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.utils.config import ensure_dir, load_yaml
from src.utils.io import write_json


os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


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
        from datasets import DatasetDict, load_dataset
    except ImportError as exc:
        raise SystemExit("Install the `datasets` package before downloading TextVQA.") from exc

    selected_splits = config.get("splits") or ["train", "validation"]
    selected = {}
    for split in selected_splits:
        print(f"Downloading split: {split}")
        try:
            selected[split] = load_dataset(config["hf_id"], split=split, cache_dir=str(cache_dir))
        except Exception as exc:
            raise SystemExit(
                f"Failed to download split `{split}` from {config['hf_id']}.\n"
                "If Hugging Face Hub returns a CAS/Xet 401 error, run:\n"
                "  export HF_HUB_DISABLE_XET=1\n"
                "If the dataset is rate-limited, also run:\n"
                "  huggingface-cli login\n"
                "or set HF_TOKEN in your environment."
            ) from exc

    if not selected:
        raise SystemExit(f"No requested splits found. Requested={selected_splits}")

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
