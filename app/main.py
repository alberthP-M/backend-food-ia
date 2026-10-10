from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import DATA_DIR
from .evaluation import evaluate
from .recommendation import RecommendationService
from .schemas import FeedbackRequest, RecommendRequest, RecommendResponse


app = FastAPI(
    title="Food AI Recommender API",
    version="1.0.0",
    description="Búsqueda y recomendación semántica de platos para un POS de restaurante.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

service: RecommendationService | None = None


def get_service() -> RecommendationService:
    global service
    if service is None:
        service = RecommendationService()
    return service


@app.get("/api/health")
def health() -> dict[str, object]:
    if service is None:
        return {
            "status": "loading",
            "service": "food-ai-recommender",
            "model_status": "not_initialized",
        }
    current = service
    return {
        "status": "ok",
        "service": "food-ai-recommender",
        "model": "Recomendador semántico con embeddings multilingües",
        "embedding_backend": current.index.backend,
        "menu_items": len(current.menu),
        "model_status": "ready",
        "ranking_weights": current.get_ranking_weights(),
    }


@app.get("/api/ready")
def ready() -> dict[str, object]:
    try:
        current = get_service()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="El modelo todavía no está disponible. Revisa los logs del backend.",
        ) from exc
    return {
        "status": "ready",
        "service": "food-ai-recommender",
        "embedding_backend": current.index.backend,
        "menu_items": len(current.menu),
        "ranking_weights": current.get_ranking_weights(),
    }


@app.get("/api/menu")
def menu() -> list[dict[str, object]]:
    return get_service().get_menu()


@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> dict[str, object]:
    try:
        return get_service().recommend(payload.query.strip(), payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/feedback")
def feedback(payload: FeedbackRequest) -> dict[str, str]:
    feedback_path = DATA_DIR / "feedback.jsonl"
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    event = payload.model_dump()
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    with feedback_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return {"status": "recorded"}


@app.get("/api/evaluation")
def evaluation() -> dict[str, object]:
    current = get_service()
    return evaluate(current, DATA_DIR / "evaluation_queries.jsonl")
