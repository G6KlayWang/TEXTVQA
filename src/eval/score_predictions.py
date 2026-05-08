from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from src.eval.llm_judge import judge_rows
from src.eval.metrics import aggregate_metrics
from src.utils.config import ensure_dir, load_yaml
from src.utils.io import read_jsonl, write_json
from src.utils.repro import config_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a TextVQA predictions JSONL file.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", default="artifacts/metrics")
    parser.add_argument("--eval_config", default="configs/eval.yaml")
    return parser.parse_args()


def scalar_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = ["tag", "n", "textvqa_accuracy", "exact_match", "f1_token", "bleu", "meteor", "rouge1", "rouge2", "rougeL"]
    return {key: metrics.get(key) for key in keys}


def update_csv(path: Path, row: dict[str, Any], key_field: str = "tag") -> None:
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    rows = [existing for existing in rows if existing.get(key_field) != row[key_field]]
    rows.append(row)
    fieldnames = list(row.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(args.predictions))
    eval_config = load_yaml(args.eval_config)
    metrics = aggregate_metrics(rows)
    judge = judge_rows(rows, eval_config)
    if judge.get("enabled"):
        metrics["llm_judge"] = judge["mean_score"]
        metrics["llm_judge_n"] = judge.get("n")
        metrics["llm_judge_batch_size"] = judge.get("batch_size")
        metrics["llm_judge_api_calls"] = judge.get("api_calls")
        metrics["llm_judge_samples"] = judge.get("samples", [])
    else:
        metrics["llm_judge"] = None
        metrics["llm_judge_status"] = judge
    metrics["tag"] = args.tag
    metrics["prediction_file"] = args.predictions
    metrics["snapshot"] = config_snapshot(eval=eval_config)

    out_dir = ensure_dir(args.out)
    write_json(metrics, out_dir / f"{args.tag}_metrics.json")
    summary = scalar_summary(metrics)
    summary["llm_judge"] = metrics.get("llm_judge")
    update_csv(out_dir / "summary.csv", summary)
    if "ablation" in args.tag:
        update_csv(out_dir / "ablation_summary.csv", summary)
    print(f"Scored {args.predictions}: accuracy={metrics['textvqa_accuracy']:.4f}")


if __name__ == "__main__":
    main()
