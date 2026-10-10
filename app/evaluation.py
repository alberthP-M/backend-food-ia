from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .recommendation import RecommendationService


def precision_at_k(recommended_ids: list[str], relevant_ids: list[str], k: int = 3) -> float:
    top = recommended_ids[:k]
    if not top:
        return 0.0
    relevant = set(relevant_ids)
    return sum(item_id in relevant for item_id in top) / k


def load_queries(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(service: RecommendationService, path: Path) -> dict[str, Any]:
    queries = load_queries(path)
    model_scores: list[float] = []
    baseline_scores: list[float] = []
    rows: list[dict[str, Any]] = []
    for row in queries:
        model = service.recommend(row["query"], top_k=3)
        baseline = service.baseline(row["query"], top_k=3)
        model_ids = [item["id"] for item in model["recommendations"]]
        baseline_ids = [item["id"] for item in baseline]
        model_score = precision_at_k(model_ids, row["relevant_ids"], 3)
        baseline_score = precision_at_k(baseline_ids, row["relevant_ids"], 3)
        model_scores.append(model_score)
        baseline_scores.append(baseline_score)
        rows.append(
            {
                "id": row["id"],
                "query": row["query"],
                "model_ids": model_ids,
                "baseline_ids": baseline_ids,
                "model_precision_at_3": model_score,
                "baseline_precision_at_3": baseline_score,
            }
        )
    return {
        "metric": "Precision@3",
        "query_count": len(queries),
        "model_backend": service.index.backend,
        "model_precision_at_3": sum(model_scores) / len(model_scores),
        "baseline_precision_at_3": sum(baseline_scores) / len(baseline_scores),
        "queries": rows,
    }
