from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def judge_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    judge_config = config.get("llm_judge") or {}
    if not judge_config.get("enabled", False):
        return {"enabled": False, "mean_score": None, "samples": []}
    if not os.environ.get("OPENAI_API_KEY"):
        return {"enabled": False, "mean_score": None, "error": "OPENAI_API_KEY is not set", "samples": []}

    try:
        from openai import OpenAI
    except ImportError:
        return {"enabled": False, "mean_score": None, "error": "openai package is not installed", "samples": []}

    template_path = judge_config.get("prompt_template", "src/eval/templates/judge_v1.txt")
    template = Path(template_path).read_text(encoding="utf-8")
    model = judge_config.get("model", "gpt-4o-mini")
    client = OpenAI()
    max_samples = judge_config.get("max_samples")
    rows_to_judge = rows if max_samples in (None, "null") else rows[: int(max_samples)]
    judged = []
    for row in rows_to_judge:
        prompt = template.format(
            question=row.get("question", ""),
            answers=", ".join(row.get("answers") or []),
            prediction=row.get("prediction", ""),
        )
        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=0,
        )
        text = response.output_text.strip()
        try:
            parsed = json.loads(text)
            score = float(parsed.get("score", 0.0))
            reason = parsed.get("reason", "")
        except Exception:
            score = 0.0
            reason = text
        judged.append({"id": row.get("id"), "score": max(0.0, min(score, 1.0)), "reason": reason})
    mean = sum(item["score"] for item in judged) / len(judged) if judged else None
    return {"enabled": True, "model": model, "mean_score": mean, "samples": judged}
