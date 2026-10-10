from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ARTIFACT_DIR, DATA_DIR
from app.evaluation import evaluate
from app.recommendation import RecommendationService


def candidate_weights() -> list[dict[str, float]]:
    candidates = []
    for semantic in (0.50, 0.55, 0.60, 0.65, 0.70):
        for structured in (0.15, 0.20, 0.25, 0.30, 0.35):
            popularity = round(1.0 - semantic - structured, 2)
            if 0.05 <= popularity <= 0.25:
                candidates.append(
                    {
                        "semantic": semantic,
                        "structured": structured,
                        "popularity": popularity,
                    }
                )
    return candidates


if __name__ == "__main__":
    service = RecommendationService()
    results = []
    for weights in candidate_weights():
        evaluation = evaluate(
            service,
            DATA_DIR / "evaluation_queries.jsonl",
            split="validation",
            weights=weights,
        )
        results.append(
            {
                "weights": weights,
                "precision_at_3": evaluation["model_precision_at_3"],
            }
        )
    results.sort(key=lambda row: row["precision_at_3"], reverse=True)
    output = {
        "selection_split": "validation",
        "metric": "Precision@3",
        "candidates": results,
        "selected_weights": results[0]["weights"],
        "selected_precision_at_3": results[0]["precision_at_3"],
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "weight_search.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
