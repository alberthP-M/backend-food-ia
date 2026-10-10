from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"loading", "ok"}


def test_ready_endpoint_initializes_model() -> None:
    response = client.get("/api/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["menu_items"] == 80


def test_recommend_endpoint_returns_top_three() -> None:
    response = client.post(
        "/api/recommend",
        json={"query": "vegetariano sin lactosa y poco picante", "top_k": 3, "channel": "kiosk"},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["recommendations"]) <= 3
    assert body["parsed_constraints"]["diets"]


def test_invalid_query_is_rejected() -> None:
    response = client.post("/api/recommend", json={"query": "x"})
    assert response.status_code == 422
