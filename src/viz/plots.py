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


TAG_LABELS = {
    "zero_shot": "Qwen zero-shot",
    "finetuned": "Qwen full LoRA",
    "ablation_lora_attn_only": "Qwen attn-only LoRA",
    "ablation_ocr_prompt": "Qwen OCR prompt",
    "ablation_ocr_hint": "Qwen OCR hint",
    "ablation_highres": "Qwen high-resolution",
    "blip2_zero_shot": "BLIP-2 zero-shot",
    "blip2_finetuned": "BLIP-2 full LoRA",
    "blip2_ablation_qformer_only": "BLIP-2 Q-Former-only",
    "blip2_ablation_ocr_hint": "BLIP-2 OCR hint",
}


def _family(tag: str) -> str:
    return "BLIP-2" if tag.startswith("blip2") else "Qwen"


def plot_accuracy(rows: list[dict], output: Path, title: str) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots.")
        return
    family_colors = {"Qwen": "#2f6f6d", "BLIP-2": "#d97706"}
    enriched = [
        {
            "tag": row["tag"],
            "label": TAG_LABELS.get(row["tag"], row["tag"]),
            "value": float(row.get("textvqa_accuracy") or 0.0),
            "family": _family(row["tag"]),
        }
        for row in rows
    ]
    enriched.sort(key=lambda item: item["value"], reverse=True)
    labels = [item["label"] for item in enriched]
    values = [item["value"] for item in enriched]
    colors = [family_colors[item["family"]] for item in enriched]
    fig, ax = plt.subplots(figsize=(7, max(3.5, len(enriched) * 0.45)))
    bars = ax.barh(labels, values, color=colors, edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("TextVQA soft accuracy")
    ax.set_xlim(0, max(1.0, max(values) * 1.18 if values else 1.0))
    ax.set_title(title)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=family_colors["Qwen"]),
        plt.Rectangle((0, 0), 1, 1, color=family_colors["BLIP-2"]),
    ]
    ax.legend(handles, ["Qwen2.5-VL", "BLIP-2"], loc="lower right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
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
