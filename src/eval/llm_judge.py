from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        score = 0.0
    return max(0.0, min(score, 1.0))


def build_batch_prompt(batch: list[dict[str, Any]]) -> str:
    items = []
    for local_id, row in enumerate(batch):
        items.append(
            {
                "local_id": local_id,
                "question": row.get("question", ""),
                "ground_truth_answers": row.get("answers") or [],
                "prediction": row.get("prediction", ""),
            }
        )
    payload = json.dumps(items, ensure_ascii=False)
    return (
        "You are judging TextVQA model predictions. For each item, decide whether the "
        "prediction is semantically equivalent to any ground-truth answer for the question. "
        "Return only a JSON array. Each array item must have keys local_id, score, and reason. "
        "The score must be a number from 0 to 1, where 1 means fully correct/equivalent and "
        "0 means incorrect. Keep each reason brief.\n\n"
        f"Items:\n{payload}"
    )


def parse_batch_response(text: str, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(text)
    except Exception:
        return [
            {
                "id": row.get("id"),
                "score": 0.0,
                "reason": f"Could not parse judge response: {text[:300]}",
            }
            for row in batch
        ]
    if not isinstance(parsed, list):
        parsed = parsed.get("items", []) if isinstance(parsed, dict) else []

    by_local_id = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            local_id = int(item.get("local_id"))
        except Exception:
            continue
        by_local_id[local_id] = item

    judged = []
    for local_id, row in enumerate(batch):
        item = by_local_id.get(local_id, {})
        judged.append(
            {
                "id": row.get("id"),
                "score": clamp_score(item.get("score", 0.0)),
                "reason": str(item.get("reason", "")),
            }
        )
    return judged


def judge_batch(client: Any, model: str, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    response = client.responses.create(
        model=model,
        input=build_batch_prompt(batch),
        temperature=0,
    )
    return parse_batch_response(response.output_text.strip(), batch)


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

    model = judge_config.get("model", "gpt-4o-mini")
    client = OpenAI()
    max_samples = judge_config.get("max_samples")
    rows_to_judge = rows if max_samples in (None, "null") else rows[: int(max_samples)]
    batch_size = max(1, int(judge_config.get("batch_size", 1)))
    max_workers = max(1, int(judge_config.get("max_workers", 1)))
    show_progress = bool(judge_config.get("show_progress", True))
    progress_label = str(judge_config.get("progress_label", "llm-judge"))
    judged = []
    batches = chunked(rows_to_judge, batch_size)
    api_calls = len(batches)
    progress_bar = None
    if show_progress:
        try:
            from tqdm import tqdm

            progress_bar = tqdm(total=len(batches), desc=progress_label, unit="batch", dynamic_ncols=True)
        except ImportError:
            progress_bar = None

    try:
        if max_workers == 1:
            for batch in batches:
                judged.extend(judge_batch(client, model, batch))
                if progress_bar is not None:
                    progress_bar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(judge_batch, client, model, batch) for batch in batches]
                for future in as_completed(futures):
                    judged.extend(future.result())
                    if progress_bar is not None:
                        progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    mean = sum(item["score"] for item in judged) / len(judged) if judged else None
    return {
        "enabled": True,
        "model": model,
        "mean_score": mean,
        "n": len(judged),
        "batch_size": batch_size,
        "max_workers": max_workers,
        "api_calls": api_calls,
        "samples": judged,
    }
