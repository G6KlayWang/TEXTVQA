from __future__ import annotations

import argparse
import html
from pathlib import Path

from src.eval.metrics import exact_match
from src.utils.config import ensure_dir
from src.utils.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an HTML qualitative gallery.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", default="artifacts/qualitative")
    parser.add_argument("--max_examples", type=int, default=40)
    return parser.parse_args()


def render_card(row: dict) -> str:
    image_path = html.escape(row.get("image_path", ""))
    question = html.escape(row.get("question", ""))
    prediction = html.escape(str(row.get("prediction", "")))
    answers = html.escape(", ".join(row.get("answers") or []))
    status = "correct" if exact_match(row.get("prediction", ""), row.get("answers") or []) else "error"
    return f"""
    <article class="card {status}">
      <img src="../../{image_path}" alt="">
      <div><strong>Question</strong><p>{question}</p></div>
      <div><strong>Prediction</strong><p>{prediction}</p></div>
      <div><strong>Answers</strong><p>{answers}</p></div>
    </article>
    """


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(args.predictions))
    correct = [row for row in rows if exact_match(row.get("prediction", ""), row.get("answers") or [])]
    errors = [row for row in rows if not exact_match(row.get("prediction", ""), row.get("answers") or [])]
    half = max(args.max_examples // 2, 1)
    selected = correct[:half] + errors[:half]
    cards = "\n".join(render_card(row) for row in selected)
    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>TextVQA Qualitative Examples</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #202124; }}
    h1 {{ font-size: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #dadce0; border-radius: 8px; padding: 12px; }}
    .card.correct {{ border-left: 6px solid #2e7d32; }}
    .card.error {{ border-left: 6px solid #b3261e; }}
    img {{ width: 100%; height: 180px; object-fit: contain; background: #f8f9fa; }}
    p {{ margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>TextVQA Qualitative Examples</h1>
  <div class="grid">{cards}</div>
</body>
</html>
"""
    out_dir = ensure_dir(args.out)
    path = Path(out_dir) / "gallery.html"
    path.write_text(html_doc, encoding="utf-8")
    print(f"Wrote qualitative gallery to {path}")


if __name__ == "__main__":
    main()

