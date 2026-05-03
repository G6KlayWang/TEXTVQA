from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.utils.config import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate result plots from metrics CSV files.")
    parser.add_argument("--metrics_dir", default="artifacts/metrics")
    parser.add_argument("--out", default="artifacts/figures")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_accuracy(rows: list[dict], output: Path, title: str) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots.")
        return
    labels = [row["tag"] for row in rows]
    values = [float(row.get("textvqa_accuracy") or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 1.5), 4))
    ax.bar(labels, values, color="#2f6f6d")
    ax.set_ylabel("TextVQA soft accuracy")
    ax.set_ylim(0, max(1.0, max(values) * 1.15 if values else 1.0))
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_per_category(metrics_dir: Path, output: Path) -> None:
    metrics = read_json(metrics_dir / "finetuned_metrics.json")
    if not metrics:
        candidates = sorted(metrics_dir.glob("*_metrics.json"))
        if candidates:
            metrics = read_json(candidates[0])
    per_type = metrics.get("per_question_type") or {}
    if not per_type:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping per-category plot.")
        return
    items = sorted(per_type.items(), key=lambda item: item[1].get("n", 0), reverse=True)[:12]
    labels = [key for key, _ in items]
    values = [float(value.get("textvqa_accuracy") or 0.0) for _, value in items]
    fig, ax = plt.subplots(figsize=(max(7, len(items) * 0.8), 4))
    ax.bar(labels, values, color="#6a5acd")
    ax.set_ylabel("TextVQA soft accuracy")
    ax.set_title("Per-Question-Type Accuracy")
    ax.set_ylim(0, max(1.0, max(values) * 1.15 if values else 1.0))
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def find_trainer_state() -> Path | None:
    candidates = [
        Path("artifacts/checkpoints/lora-qwen25vl/trainer_state.json"),
        *Path("artifacts/checkpoints/lora-qwen25vl").glob("checkpoint-*/trainer_state.json"),
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    return sorted(existing, key=lambda path: path.stat().st_mtime)[-1]


def plot_training_curve(output: Path) -> None:
    state_path = find_trainer_state()
    if state_path is None:
        return
    state = read_json(state_path)
    history = state.get("log_history") or []
    train = [(item.get("step"), item.get("loss")) for item in history if item.get("loss") is not None]
    eval_loss = [(item.get("step"), item.get("eval_loss")) for item in history if item.get("eval_loss") is not None]
    if not train and not eval_loss:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping training curve.")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    if train:
        ax.plot([x for x, _ in train], [y for _, y in train], label="train loss")
    if eval_loss:
        ax.plot([x for x, _ in eval_loss], [y for _, y in eval_loss], label="eval loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("LoRA Training Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    out_dir = ensure_dir(args.out)
    plot_accuracy(read_csv(metrics_dir / "summary.csv"), out_dir / "accuracy_comparison.pdf", "TextVQA Accuracy")
    plot_accuracy(read_csv(metrics_dir / "ablation_summary.csv"), out_dir / "ablation_accuracy.pdf", "Ablation Accuracy")
    plot_per_category(metrics_dir, out_dir / "per_category_acc.pdf")
    plot_training_curve(out_dir / "training_curve.pdf")
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
