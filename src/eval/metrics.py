from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


ARTICLES = {"a", "an", "the"}
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.lower().strip()
    text = text.translate(PUNCT_TABLE)
    tokens = [token for token in text.split() if token not in ARTICLES]
    return " ".join(tokens)


def textvqa_soft_accuracy(prediction: str, answers: list[str]) -> float:
    pred = normalize_answer(prediction)
    matches = sum(1 for answer in answers if normalize_answer(answer) == pred)
    return min(matches / 3.0, 1.0)


def exact_match(prediction: str, answers: list[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(normalize_answer(answer) == pred for answer in answers))


def token_f1(prediction: str, answers: list[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    pred_counts = Counter(pred_tokens)
    for answer in answers:
        ans_tokens = normalize_answer(answer).split()
        if not ans_tokens:
            continue
        ans_counts = Counter(ans_tokens)
        overlap = sum((pred_counts & ans_counts).values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ans_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def question_type(question: str) -> str:
    question = question.lower().strip()
    first = re.split(r"\s+", question)[0] if question else "unknown"
    return first.rstrip(string.punctuation) or "unknown"


def compute_bleu(predictions: list[str], references: list[list[str]]) -> float | None:
    try:
        import sacrebleu
    except ImportError:
        return None
    refs = [[answers[0] if answers else "" for answers in references]]
    return float(sacrebleu.corpus_bleu(predictions, refs).score)


def compute_meteor(predictions: list[str], references: list[list[str]]) -> float | None:
    try:
        from nltk.translate.meteor_score import meteor_score
    except ImportError:
        return None
    scores = []
    for pred, answers in zip(predictions, references):
        refs = [normalize_answer(answer).split() for answer in answers] or [[]]
        scores.append(meteor_score(refs, normalize_answer(pred).split()))
    return sum(scores) / len(scores) if scores else None


def compute_rouge(predictions: list[str], references: list[list[str]]) -> dict[str, float] | None:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return None
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, answers in zip(predictions, references):
        reference = answers[0] if answers else ""
        scores = scorer.score(reference, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure
    n = max(len(predictions), 1)
    return {key: value / n for key, value in totals.items()}


def aggregate_metrics(rows: list[dict[str, Any]], include_semantic: bool = True) -> dict[str, Any]:
    predictions = [str(row.get("prediction", "")) for row in rows]
    references = [list(row.get("answers") or []) for row in rows]
    soft_scores = [textvqa_soft_accuracy(pred, answers) for pred, answers in zip(predictions, references)]
    exact_scores = [exact_match(pred, answers) for pred, answers in zip(predictions, references)]
    f1_scores = [token_f1(pred, answers) for pred, answers in zip(predictions, references)]
    metrics: dict[str, Any] = {
        "n": len(rows),
        "textvqa_accuracy": sum(soft_scores) / len(soft_scores) if soft_scores else 0.0,
        "exact_match": sum(exact_scores) / len(exact_scores) if exact_scores else 0.0,
        "f1_token": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
    }
    if include_semantic:
        metrics["bleu"] = compute_bleu(predictions, references)
        metrics["meteor"] = compute_meteor(predictions, references)
        rouge = compute_rouge(predictions, references)
        if rouge is None:
            metrics["rouge"] = None
        else:
            metrics.update(rouge)

    by_type: dict[str, list[float]] = {}
    for row, score in zip(rows, soft_scores):
        by_type.setdefault(question_type(row.get("question", "")), []).append(score)
    metrics["per_question_type"] = {
        key: {"n": len(values), "textvqa_accuracy": sum(values) / len(values)}
        for key, values in sorted(by_type.items())
    }
    return metrics

