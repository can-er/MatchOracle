"""Sprint 00 DoD: the app boots and /health answers 200 OK."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "matchoracle"
    assert "version" in body


def test_app_imports_and_has_routes() -> None:
    paths = {route.path for route in app.routes}  # type: ignore[attr-defined]
    assert "/health" in paths
