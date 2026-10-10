from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .catalog import load_menu
from .config import (
    ARTIFACT_DIR,
    DATA_DIR,
    DEFAULT_RANKING_WEIGHTS,
    MODEL_BACKEND,
    MODEL_NAME,
)
from .parser import QueryIntent, normalize, parse_query


class EmbeddingIndex:
    """Semantic index with a pretrained multilingual encoder and a test fallback."""

    def __init__(
        self,
        menu: list[dict[str, object]],
        artifact_dir: Path = ARTIFACT_DIR,
        backend: str = MODEL_BACKEND,
    ) -> None:
        self.menu = menu
        self.artifact_dir = artifact_dir
        self.requested_backend = backend
        self.model_name = MODEL_NAME
        self.backend = "tfidf"
        self.encoder: Any = None
        self.vectorizer: TfidfVectorizer | None = None
        self.embeddings: np.ndarray
        self._load_encoder()
        self._fit_index()

    def _load_encoder(self) -> None:
        if self.requested_backend == "tfidf":
            return
        if self.requested_backend in {"auto", "fastembed"}:
            try:
                from fastembed import TextEmbedding

                self.encoder = TextEmbedding(model_name=self.model_name)
                self.backend = "fastembed"
                return
            except Exception as exc:  # pragma: no cover - depends on model cache
                if self.requested_backend == "fastembed":
                    raise RuntimeError(
                        "No se pudo cargar el modelo FastEmbed. "
                        "Verifica la conexión o usa MODEL_BACKEND=tfidf para pruebas."
                    ) from exc
        try:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(self.model_name)
            self.backend = "sentence-transformers"
        except Exception as exc:  # pragma: no cover - depends on network/model cache
            if self.requested_backend == "sentence-transformers":
                raise RuntimeError(
                    "No se pudo cargar el modelo de embeddings. "
                    "Verifica la conexión o usa MODEL_BACKEND=tfidf para pruebas."
                ) from exc
            self.backend = "tfidf"

    @staticmethod
    def item_text(item: dict[str, object]) -> str:
        return " ".join(
            [
                str(item["name"]),
                str(item["description"]),
                str(item["category"]),
                " ".join(item["diet_tags"]),
                " ".join(item["ingredients"]),
                str(item["spice_level"]),
                f"{item['calories']} calorias",
                f"{item['protein_g']} gramos de proteina",
            ]
        )

    def _fit_index(self) -> None:
        texts = [self.item_text(item) for item in self.menu]
        if self.backend == "fastembed":
            vectors = self.encoder.embed([self._embedding_text(text, is_query=False) for text in texts])
            self.embeddings = np.asarray(list(vectors), dtype=np.float32)
            self.embeddings = self.embeddings / np.maximum(
                np.linalg.norm(self.embeddings, axis=1, keepdims=True), 1e-12
            )
            return
        if self.backend == "sentence-transformers":
            self.embeddings = np.asarray(
                self.encoder.encode(texts, normalize_embeddings=True), dtype=np.float32
            )
        else:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")
            self.embeddings = self.vectorizer.fit_transform(texts).toarray().astype(np.float32)

    def encode(self, text: str) -> np.ndarray:
        if self.backend == "fastembed":
            vector = np.asarray(
                list(self.encoder.embed([self._embedding_text(text, is_query=True)]))[0],
                dtype=np.float32,
            )
            return vector / max(float(np.linalg.norm(vector)), 1e-12)
        if self.backend == "sentence-transformers":
            return np.asarray(
                self.encoder.encode([text], normalize_embeddings=True)[0], dtype=np.float32
            )
        assert self.vectorizer is not None
        return self.vectorizer.transform([text]).toarray()[0].astype(np.float32)

    def _embedding_text(self, text: str, is_query: bool) -> str:
        """Apply E5 prefixes only when an E5 model is selected."""
        if "e5" in self.model_name.lower():
            return f"{'query' if is_query else 'passage'}: {text}"
        return text

    def save(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.artifact_dir / "menu_embeddings.npy", self.embeddings)
        metadata = {
            "backend": self.backend,
            "model_name": self.model_name,
            "menu_ids": [item["id"] for item in self.menu],
            "embedding_dimension": int(self.embeddings.shape[1]),
        }
        (self.artifact_dir / "index_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )


class RecommendationService:
    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        artifact_dir: Path = ARTIFACT_DIR,
        backend: str = MODEL_BACKEND,
    ) -> None:
        self.data_dir = data_dir
        self.menu = load_menu(data_dir / "menu.csv")
        self.index = EmbeddingIndex(self.menu, artifact_dir=artifact_dir, backend=backend)

    @staticmethod
    def _normalize_weights(weights: dict[str, float] | None = None) -> dict[str, float]:
        selected = dict(DEFAULT_RANKING_WEIGHTS if weights is None else weights)
        if set(selected) != {"semantic", "structured", "popularity"}:
            raise ValueError("Los pesos deben incluir semantic, structured y popularity.")
        if any(value < 0 for value in selected.values()):
            raise ValueError("Los pesos no pueden ser negativos.")
        total = sum(selected.values())
        if total <= 0:
            raise ValueError("La suma de los pesos debe ser mayor que cero.")
        return {key: round(value / total, 4) for key, value in selected.items()}

    @staticmethod
    def _diet_compatible(item: dict[str, object], diets: list[str]) -> bool:
        tags = set(item["diet_tags"])
        for diet in diets:
            if diet == "vegetarian" and "vegetarian" not in tags:
                return False
            if diet == "vegan" and "vegan" not in tags:
                return False
            if diet == "gluten_free" and "gluten_free" not in tags:
                return False
            if diet == "dairy_free" and "dairy_free" not in tags:
                return False
            if diet == "low_calorie" and "low_calorie" not in tags:
                return False
            if diet == "high_protein" and "high_protein" not in tags:
                return False
        return True

    @staticmethod
    def _spice_compatible(item_level: str, requested: str | None) -> bool:
        if requested is None:
            return True
        levels = {"low": 0, "medium": 1, "high": 2}
        return levels.get(item_level, 1) <= levels.get(requested, 2)

    def filter_candidates(
        self, intent: QueryIntent, ignore_soft_constraints: bool = False
    ) -> list[dict[str, object]]:
        candidates = []
        excluded = set(intent.excluded_allergens)
        main_meal_context = not intent.category_hints and bool(
            intent.diets
            or intent.excluded_allergens
            or intent.protein
            or intent.spice_level
            or intent.max_price is not None
        )
        for item in self.menu:
            if not item["available"]:
                continue
            if excluded.intersection(item["allergens"]):
                continue
            if main_meal_context and item["category"] in {
                "bebidas",
                "postres",
                "acompanamientos",
            }:
                continue
            if not ignore_soft_constraints:
                if not self._diet_compatible(item, intent.diets):
                    continue
                if intent.max_price is not None and item["price"] > intent.max_price:
                    continue
                if intent.min_price is not None and item["price"] < intent.min_price:
                    continue
                if not self._spice_compatible(item["spice_level"], intent.spice_level):
                    continue
                if intent.protein and not any(
                    intent.protein in normalize(value)
                    for value in [item["name"], *item["ingredients"]]
                ):
                    continue
                if intent.category_hints and item["category"] not in intent.category_hints:
                    continue
            candidates.append(item)
        return candidates

    @staticmethod
    def _has_hard_conflict(intent: QueryIntent) -> bool:
        protein_allergen = {
            "pescado": "fish",
            "salmon": "fish",
            "camarones": "shellfish",
        }
        return bool(
            intent.protein
            and protein_allergen.get(intent.protein) in set(intent.excluded_allergens)
        )

    @staticmethod
    def _public_item(item: dict[str, object]) -> dict[str, object]:
        return {
            "id": item["id"],
            "name": item["name"],
            "description": item["description"],
            "category": item["category"],
            "price": item["price"],
            "ingredients": item["ingredients"],
            "diet_tags": item["diet_tags"],
            "allergens": item["allergens"],
            "available": item["available"],
            "prep_minutes": item["prep_minutes"],
        }

    @staticmethod
    def _explanation(item: dict[str, object], intent: QueryIntent) -> str:
        reasons: list[str] = []
        if intent.diets:
            reasons.append("cumple tus preferencias dietéticas")
        if intent.excluded_allergens:
            reasons.append("no coincide con los alérgenos excluidos")
        if intent.max_price is not None:
            reasons.append(f"está dentro de Bs {intent.max_price:.0f}")
        if intent.spice_level == "low":
            reasons.append("tiene un nivel de picante bajo")
        if intent.protein:
            reasons.append(f"incluye {intent.protein}")
        if not reasons:
            reasons.append("es semánticamente relevante para tu solicitud")
        return "Te lo recomendamos porque " + ", ".join(reasons) + "."

    def _to_recommendation(
        self, item: dict[str, object], score: float, intent: QueryIntent
    ) -> dict[str, object]:
        result = self._public_item(item)
        result.update(
            {
                "score": round(float(max(0.0, min(1.0, score))), 4),
                "explanation": self._explanation(item, intent),
            }
        )
        return result

    @staticmethod
    def _structured_match_score(item: dict[str, object], intent: QueryIntent) -> float:
        score = 0.0
        if intent.diets:
            score += 0.20
        if intent.excluded_allergens:
            score += 0.15
        if intent.category_hints and item["category"] in intent.category_hints:
            score += 0.35
        if intent.protein and any(
            intent.protein in normalize(value)
            for value in [item["name"], *item["ingredients"]]
        ):
            score += 0.20
        if intent.spice_level and item["spice_level"] == intent.spice_level:
            score += 0.05
        if intent.max_price is not None and item["price"] <= intent.max_price:
            score += 0.05
        return min(score, 1.0)

    def baseline(self, query: str, top_k: int = 3) -> list[dict[str, object]]:
        intent = parse_query(query)
        candidates = sorted(
            self.filter_candidates(intent),
            key=lambda item: item["popularity"],
            reverse=True,
        )
        return [self._to_recommendation(item, 0.0, intent) for item in candidates[:top_k]]

    def recommend(
        self,
        query: str,
        top_k: int = 3,
        weights: dict[str, float] | None = None,
    ) -> dict[str, object]:
        intent = parse_query(query)
        ranking_weights = self._normalize_weights(weights)
        safety_notice = (
            "Verifica ingredientes y contaminación cruzada con el personal. "
            "Un alérgeno no registrado no garantiza que el plato sea seguro."
        )
        candidates = self.filter_candidates(intent)
        fallback_used = False
        if not candidates and not self._has_hard_conflict(intent):
            candidates = self.filter_candidates(intent, ignore_soft_constraints=True)
            fallback_used = True

        query_embedding = self.index.encode(intent.semantic_query)
        if not candidates:
            return {
                "query": query,
                "parsed_constraints": intent.as_dict(),
                "model": "Recomendador semántico con embeddings multilingües",
                "backend": self.index.backend,
                "ranking_weights": ranking_weights,
                "recommendations": [],
                "alternatives": [],
                "message": "No encontramos platos disponibles compatibles con tus restricciones de seguridad.",
                "safety_notice": safety_notice,
            }

        candidate_indexes = [self.menu.index(item) for item in candidates]
        matrix = self.index.embeddings[candidate_indexes]
        similarities = cosine_similarity(query_embedding.reshape(1, -1), matrix)[0]
        popularity = np.array([float(item["popularity"]) for item in candidates])
        popularity = popularity / max(float(popularity.max()), 1.0)
        structured = np.array(
            [self._structured_match_score(item, intent) for item in candidates]
        )
        scores = (
            ranking_weights["semantic"] * similarities
            + ranking_weights["structured"] * structured
            + ranking_weights["popularity"] * popularity
        )
        order = np.argsort(-scores)[:top_k]
        recommendations = [
            self._to_recommendation(candidates[index], float(scores[index]), intent)
            for index in order
        ]
        message = None
        if fallback_used:
            message = (
                "No encontramos una coincidencia exacta. Mostramos alternativas compatibles "
                "con tus restricciones de seguridad; revisa las condiciones destacadas."
            )
        return {
            "query": query,
            "parsed_constraints": intent.as_dict(),
            "model": "Recomendador semántico con embeddings multilingües",
            "backend": self.index.backend,
            "ranking_weights": ranking_weights,
            "recommendations": recommendations,
            "alternatives": [],
            "message": message,
            "safety_notice": safety_notice,
        }

    def get_ranking_weights(self) -> dict[str, float]:
        return self._normalize_weights()

    def get_menu(self) -> list[dict[str, object]]:
        return [self._public_item(item) for item in self.menu]
