"""API tests for the Sprint 06 REST surface (predict, history, agents, connectors).

Uses an isolated in-memory SQLite database injected via FastAPI dependency
override — no Postgres/Redis/LLM key required.
"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import models  # noqa: F401 — register models on the metadata
from app.db.base import Base, get_session
from app.main import app

_engine = create_engine(
    "sqlite+pysqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_TestingSession = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
Base.metadata.create_all(_engine)


def _override_session() -> Iterator[Session]:
    session = _TestingSession()
    try:
        yield session
        session.commit()
    finally:
        session.close()


app.dependency_overrides[get_session] = _override_session
client = TestClient(app)


def test_predict_then_fetch_history_and_agents() -> None:
    resp = client.post("/api/v1/predict", json={"entity": "Acme Corp", "domain": "finance"})
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["entity"] == "Acme Corp"
    assert body["prediction"] in {"Positive", "Negative"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["explanation"]
    prediction_id = body["id"]

    listing = client.get("/api/v1/predictions").json()
    assert listing["total"] >= 1
    assert any(item["id"] == prediction_id for item in listing["items"])

    agents = client.get(f"/api/v1/predictions/{prediction_id}/agents").json()
    assert len(agents) >= 6
    assert {a["agent_name"] for a in agents} >= {"historical", "trend", "expert"}


def test_get_unknown_prediction_returns_404() -> None:
    resp = client.get("/api/v1/predictions/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


def test_connectors_create_list_and_conflict() -> None:
    created = client.post(
        "/api/v1/connectors",
        json={"name": "Stock API", "type": "rest", "configuration": {"base_url": "https://x"}},
    )
    assert created.status_code == 201, created.text
    assert created.json()["name"] == "Stock API"

    names = [c["name"] for c in client.get("/api/v1/connectors").json()]
    assert "Stock API" in names

    dup = client.post("/api/v1/connectors", json={"name": "Stock API", "type": "rest"})
    assert dup.status_code == 409


def test_worldcup_group_endpoint_404_without_data() -> None:
    # Offline (no football-data key / network): the group can't be fetched -> 404.
    resp = client.get("/api/v1/worldcup/groups/A")
    assert resp.status_code == 404


def test_worldcup_matchday_endpoint_404_without_data() -> None:
    resp = client.get("/api/v1/worldcup/matchday/1")
    assert resp.status_code == 404


def test_worldcup_accuracy_endpoint_empty() -> None:
    # No finished snapshots in the test DB -> a valid, empty report.
    resp = client.get("/api/v1/worldcup/accuracy")
    assert resp.status_code == 200
    assert resp.json()["evaluated"] == 0


def test_worldcup_champion_endpoint_404_without_data() -> None:
    resp = client.get("/api/v1/worldcup/champion")
    assert resp.status_code == 404


def test_outcome_benchmark_and_weights_flow() -> None:
    # Predict, then record the real outcome, then inspect benchmark + weights.
    resp = client.post("/api/v1/predict", json={"entity": "Benchmark Co", "domain": "finance"})
    assert resp.status_code == 201, resp.text
    prediction_id = resp.json()["id"]

    recorded = client.post(
        f"/api/v1/predictions/{prediction_id}/outcome",
        json={"actual": "Positive"},
    )
    assert recorded.status_code == 201, recorded.text
    assert recorded.json()["prediction_id"] == prediction_id
    assert isinstance(recorded.json()["correct"], bool)

    benchmark = client.get("/api/v1/agents/benchmark")
    assert benchmark.status_code == 200
    assert benchmark.json()["evaluated"] >= 1

    accuracy = client.get("/api/v1/agents/accuracy").json()
    assert all(0.0 <= v <= 1.0 for v in accuracy.values())

    weights = client.get("/api/v1/weights").json()["weights"]
    assert abs(sum(weights.values()) - 1.0) < 0.05

    recalc = client.post("/api/v1/weights/recalculate").json()
    assert "weights" in recalc
    assert abs(sum(recalc["weights"].values()) - 1.0) < 0.05


def test_feedback_learn_and_rollback_flow() -> None:
    resp = client.post("/api/v1/predict", json={"entity": "Feedback Co", "domain": "finance"})
    prediction_id = resp.json()["id"]

    fb = client.post(
        f"/api/v1/predictions/{prediction_id}/feedback",
        json={"verdict": "approve", "validator": "qa"},
    )
    assert fb.status_code == 201, fb.text
    assert fb.json()["reward"] == 1.0

    learn = client.post("/api/v1/weights/learn-from-feedback").json()
    assert "weights" in learn
    assert abs(sum(learn["weights"].values()) - 1.0) < 0.05

    rolled = client.post("/api/v1/weights/rollback").json()["weights"]
    assert abs(sum(rolled.values()) - 1.0) < 0.05


def test_feedback_rejects_unknown_verdict() -> None:
    resp = client.post("/api/v1/predict", json={"entity": "Bad Verdict Co", "domain": "finance"})
    pid = resp.json()["id"]
    bad = client.post(f"/api/v1/predictions/{pid}/feedback", json={"verdict": "maybe"})
    assert bad.status_code == 422  # schema validation rejects it


def test_llm_router_endpoint_reflects_policy() -> None:
    resp = client.get("/api/v1/llm/router", params={"complexity": 0.9})
    assert resp.status_code == 200
    body = resp.json()
    assert body["policy"] in {"cost", "quality", "balanced"}
    assert body["selected_tier"] in {t["name"] for t in body["tiers"]}


def test_record_outcome_unknown_prediction_404() -> None:
    resp = client.post(
        "/api/v1/predictions/00000000-0000-0000-0000-000000000000/outcome",
        json={"actual": "Positive"},
    )
    assert resp.status_code == 404


def test_worldcup_calibration_endpoint_defaults_to_seed() -> None:
    from app.prediction.score import BASE_GOALS

    resp = client.get("/api/v1/worldcup/calibration")
    assert resp.status_code == 200
    body = resp.json()
    assert body["calibrated"] is False
    assert body["samples"] == 0
    assert body["base_goals"] == BASE_GOALS  # dormant -> seed value
