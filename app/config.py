from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", ROOT_DIR / "artifacts"))
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "auto").lower()
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
