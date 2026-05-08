from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.eval.llm_judge import judge_rows
from src.utils.config import ensure_dir
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
        description="Run batched LLM-as-a-Judge on existing TextVQA prediction JSONL files."
    )
    parser.add_argument(
        "--prediction",
        action="append",
        type=parse_prediction_arg,
        required=True,
        help="Prediction file in tag=path format. Repeat for multiple model runs.",
    )
    parser.add_argument("--out", default="artifacts/llm_judge")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap per prediction file. Omit to judge every row.",
    )
    return parser.parse_args()


def write_summary(path: Path, rows: list[dict]) -> None:
    fieldnames = ["tag", "prediction_file", "n", "mean_score", "model", "batch_size", "max_workers", "api_calls"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be positive")
    if args.max_workers <= 0:
        raise SystemExit("--max_workers must be positive")

    out_dir = ensure_dir(args.out)
    summary_rows = []
    for tag, prediction_path in args.prediction:
        if not prediction_path.exists():
            print(f"Skipping missing prediction file: {prediction_path}")
            continue

        rows = list(read_jsonl(prediction_path))
        judge_config = {
            "llm_judge": {
                "enabled": True,
                "model": args.model,
                "max_samples": args.max_samples,
                "batch_size": args.batch_size,
                "max_workers": args.max_workers,
                "show_progress": True,
                "progress_label": tag,
            }
        }
        result = judge_rows(rows, judge_config)
        result["tag"] = tag
        result["prediction_file"] = str(prediction_path)
        write_json(result, out_dir / f"{tag}_llm_judge.json")

        summary_rows.append(
            {
                "tag": tag,
                "prediction_file": str(prediction_path),
                "n": result.get("n"),
                "mean_score": result.get("mean_score"),
                "model": result.get("model"),
                "batch_size": result.get("batch_size"),
                "max_workers": result.get("max_workers"),
                "api_calls": result.get("api_calls"),
            }
        )
        print(
            f"Judged {tag}: n={result.get('n')} mean={result.get('mean_score')} "
            f"api_calls={result.get('api_calls')}"
        )

    write_summary(out_dir / "summary.csv", summary_rows)
    print(f"Wrote LLM judge outputs to {out_dir}")


if __name__ == "__main__":
    main()
