from __future__ import annotations

import argparse
import os

from src.utils.config import load_yaml


os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-cache a Hugging Face model snapshot before multi-GPU launch.")
    parser.add_argument("--model_config", default="configs/model_qwen25vl.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.model_config)
    model_id = config["model_id"]
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install `huggingface_hub` before caching model weights.") from exc

    print(f"Pre-caching model snapshot for {model_id} with HF_HUB_DISABLE_XET={os.environ.get('HF_HUB_DISABLE_XET')}")
    try:
        path = snapshot_download(
            repo_id=model_id,
            token=os.environ.get("HF_TOKEN") or None,
            allow_patterns=[
                "*.json",
                "*.jinja",
                "*.safetensors",
                "*.txt",
                "*.model",
                "*.tiktoken",
                "tokenizer*",
                "preprocessor*",
                "processor*",
                "vocab*",
                "merges*",
            ],
        )
    except Exception as exc:
        raise SystemExit(
            f"Failed to cache {model_id}.\n"
            "This is usually a Hugging Face Hub auth/download issue, not a model-code issue.\n"
            "Try these commands in the same shell, then rerun:\n"
            "  export HF_HUB_DISABLE_XET=1\n"
            "  huggingface-cli login\n"
            "or set:\n"
            "  export HF_TOKEN=your_token_here\n"
            f"Original error: {exc}"
        ) from exc
    print(f"Cached {model_id} at {path}")


if __name__ == "__main__":
    main()
