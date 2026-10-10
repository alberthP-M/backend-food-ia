from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ARTIFACT_DIR
from app.recommendation import RecommendationService


if __name__ == "__main__":
    service = RecommendationService()
    service.index.save()
    print(
        f"Index built: {len(service.menu)} items, backend={service.index.backend}, "
        f"dimension={service.index.embeddings.shape[1]}, output={ARTIFACT_DIR}"
    )

