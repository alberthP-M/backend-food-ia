from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)
    channel: Literal["kiosk", "web", "waiter"] = "web"


class FeedbackRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    dish_id: str = Field(min_length=1, max_length=80)
    action: Literal["view", "click", "cart", "purchase", "dismiss"]
    channel: Literal["kiosk", "web", "waiter"] = "web"


class Recommendation(BaseModel):
    id: str
    name: str
    description: str
    category: str
    price: float
    score: float
    explanation: str
    ingredients: list[str]
    diet_tags: list[str]
    allergens: list[str]
    available: bool
    prep_minutes: int


class RecommendResponse(BaseModel):
    query: str
    parsed_constraints: dict[str, Any]
    model: str
    backend: str
    ranking_weights: dict[str, float]
    recommendations: list[Recommendation]
    alternatives: list[Recommendation] = Field(default_factory=list)
    message: str | None = None
    safety_notice: str
