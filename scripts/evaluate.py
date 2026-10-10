from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ARTIFACT_DIR, DATA_DIR
from app.evaluation import evaluate
from app.recommendation import RecommendationService


if __name__ == "__main__":
    service = RecommendationService()
    results = evaluate(service, DATA_DIR / "evaluation_queries.jsonl")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "evaluation.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))

