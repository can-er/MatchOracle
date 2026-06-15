"""Autonomous (human-free) feedback loop from real World Cup results (Sprint WC-8).

The engine ingests finished-match scorelines for its full-agent predictions and
re-tunes the agent weights — no human verdict in the loop.
"""

from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.models import AgentResult, Outcome, Prediction
from app.orchestration.service import PredictionService
from app.orchestration.weights import _CACHE_KEY, WeightManager
from app.prediction.benchmark import label_from_scoreline, realised_label


# --------------------------------------------------------------------------- #
# Label helpers
# --------------------------------------------------------------------------- #
def test_label_from_scoreline() -> None:
    assert label_from_scoreline("2-1") == 1  # home win
    assert label_from_scoreline("0-3") == 0  # away win
    assert label_from_scoreline("1-1") is None  # draw -> skipped
    assert label_from_scoreline("bad") is None


def test_realised_label_prefers_token_then_scoreline() -> None:
    assert realised_label("Positive") == 1
    assert realised_label("2-0") == 1
    assert realised_label("0-2") == 0
    assert realised_label("1-1") is None


# --------------------------------------------------------------------------- #
# Ingestion + autonomous learning
# --------------------------------------------------------------------------- #
def _make_session():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _wc_prediction(session, home, away, agent_scores, with_agents=True) -> Prediction:
    pred = Prediction(
        entity=f"{home} vs {away}", domain="worldcup",
        prediction=f"{home} 1-0 {away}", score=0.6, confidence=0.6, contributors=[],
    )
    session.add(pred)
    session.flush()
    if with_agents:
        for agent, score in agent_scores.items():
            session.add(
                AgentResult(
                    prediction_id=pred.id, agent_name=agent, score=score,
                    confidence=0.7, weight=0.2,
                )
            )
        session.flush()
    return pred


def test_ingest_only_targets_full_agent_predictions(monkeypatch) -> None:
    session = _make_session()
    try:
        svc = PredictionService(session)
        # Full-agent prediction (learnable) + a scheduler-style snapshot (no agents).
        full = _wc_prediction(session, "Brazil", "Serbia", {"historical": 0.8})
        _wc_prediction(session, "Spain", "Japan", {}, with_agents=False)
        session.commit()

        monkeypatch.setattr(
            "app.connectors.worldcup.WorldCupConnector.finished_results",
            lambda self: {("Brazil", "Serbia"): "2-0", ("Spain", "Japan"): "1-0"},
        )
        ingested = svc.ingest_worldcup_outcomes()
        # Only the full-agent prediction gets an outcome; the snapshot is skipped.
        assert ingested == 1
        outcomes = list(session.scalars(select(Outcome)))
        assert len(outcomes) == 1
        assert outcomes[0].prediction_id == full.id
        assert outcomes[0].actual == "2-0"
    finally:
        session.close()


def test_autonomous_learn_tunes_weights_from_real_results(monkeypatch) -> None:
    session = _make_session()
    wm = WeightManager()
    wm._cache.delete(_CACHE_KEY)
    try:
        svc = PredictionService(session, weight_manager=wm)
        before = wm.current()
        # 'historical' calls the home winner right every time; 'risk' always wrong.
        results = {}
        for i in range(6):
            home, away = f"H{i}", f"A{i}"
            _wc_prediction(session, home, away, {"historical": 0.9, "risk": 0.1})
            results[(home, away)] = "2-0"  # home wins -> label 1
        session.commit()

        monkeypatch.setattr(
            "app.connectors.worldcup.WorldCupConnector.finished_results",
            lambda self: results,
        )
        summary = svc.autonomous_learn()
        assert summary["ingested"] == 6
        assert summary["adjusted"] is True
        assert summary["samples"] == 12  # 6 matches x 2 agents
        assert summary["accuracy"]["historical"] == 1.0
        assert summary["accuracy"]["risk"] == 0.0
        # The agent that tracked reality is now weighted higher — no human input.
        assert wm.current()["historical"] > before["historical"]
    finally:
        wm._cache.delete(_CACHE_KEY)
        session.close()


def test_autonomous_learn_noop_without_results(monkeypatch) -> None:
    session = _make_session()
    wm = WeightManager()
    wm._cache.delete(_CACHE_KEY)
    try:
        monkeypatch.setattr(
            "app.connectors.worldcup.WorldCupConnector.finished_results", lambda self: {}
        )
        summary = PredictionService(session, weight_manager=wm).autonomous_learn()
        assert summary["ingested"] == 0
        assert summary["adjusted"] is False
    finally:
        wm._cache.delete(_CACHE_KEY)
        session.close()
