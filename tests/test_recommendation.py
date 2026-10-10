from app.recommendation import RecommendationService
from app.evaluation import precision_at_k


def test_precision_at_k_uses_fixed_k_denominator() -> None:
    assert precision_at_k(["dish-a"], ["dish-a"], k=3) == 1 / 3


def test_recommendation_respects_allergen_and_diet_constraints() -> None:
    service = RecommendationService(backend="tfidf")
    result = service.recommend("vegetariano sin lactosa sin mani poco picante", top_k=3)

    assert result["recommendations"]
    for item in result["recommendations"]:
        assert "peanuts" not in item["allergens"]
        assert "milk" not in item["allergens"]
        assert "vegetarian" in item["diet_tags"]
        assert item["prep_minutes"] >= 0


def test_recommendation_returns_message_for_no_safe_match() -> None:
    service = RecommendationService(backend="tfidf")
    result = service.recommend("alergia a pescado y sin gluten con salmon", top_k=3)

    assert result["recommendations"] == []
    assert result["message"]


def test_baseline_is_available_for_comparison() -> None:
    service = RecommendationService(backend="tfidf")
    baseline = service.baseline("quiero algo vegetariano", top_k=3)

    assert len(baseline) == 3
    assert all("id" in item for item in baseline)
