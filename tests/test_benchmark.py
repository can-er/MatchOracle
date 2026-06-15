"""Accuracy / benchmarking / auto-weighting tests (Sprints 10 & 11).

Pure-function coverage on a known dataset, plus an end-to-end pass through the
service: record outcomes → benchmark → accuracy-driven weight recalculation.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import models  # noqa: F401 — register models
from app.db.base import Base
from app.db.models import AccuracySnapshot, AgentResult, Prediction
from app.orchestration.service import PredictionService
from app.orchestration.weights import _CACHE_KEY, WeightManager
from app.prediction.benchmark import agent_accuracy, benchmark_agents, label_from_actual


# --------------------------------------------------------------------------- #
# Pure functions
# --------------------------------------------------------------------------- #
def test_label_from_actual_mapping() -> None:
    assert label_from_actual("Positive") == 1
    assert label_from_actual("WIN") == 1
    assert label_from_actual("negative") == 0
    assert label_from_actual("away") == 0
    assert label_from_actual("2-1") is None  # scoreline -> no binary class
    assert label_from_actual(None) is None


def test_agent_accuracy_directional_hits() -> None:
    obs = [
        {"agent": "a", "score": 0.8, "confidence": 0.7, "weight": 0.3, "label": 1},  # hit
        {"agent": "a", "score": 0.2, "confidence": 0.7, "weight": 0.3, "label": 1},  # miss
        {"agent": "b", "score": 0.9, "confidence": 0.6, "weight": 0.2, "label": 1},  # hit
        {"agent": "b", "score": 0.7, "confidence": 0.6, "weight": 0.2, "label": 1},  # hit
        {"agent": "c", "score": 0.3, "confidence": 0.5, "weight": 0.1, "label": None},  # skipped
    ]
    acc = agent_accuracy(obs)
    assert acc == {"a": 0.5, "b": 1.0}
    assert "c" not in acc  # all-unlabelled agent is absent


def test_benchmark_agents_report_and_flag() -> None:
    obs = []
    # Agent 'good' is right 8/10; 'bad' is right 2/10 (-> flagged underperforming).
    for i in range(10):
        obs.append(
            {"agent": "good", "score": 0.9 if i < 8 else 0.1, "confidence": 0.8,
             "weight": 0.3, "label": 1}
        )
        obs.append(
            {"agent": "bad", "score": 0.9 if i < 2 else 0.1, "confidence": 0.8,
             "weight": 0.2, "label": 1}
        )
    report = benchmark_agents(obs)
    by_name = {b.agent: b for b in report}
    assert by_name["good"].accuracy == 0.8
    assert by_name["bad"].accuracy == 0.2
    assert by_name["bad"].flag == "underperforming"
    assert by_name["good"].flag is None
    # Report is sorted best-accuracy first.
    assert report[0].agent == "good"
    # Calibration gap: confidence 0.8 vs accuracy 0.8 -> ~0 for 'good'.
    assert by_name["good"].calibration_error == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Service: record outcome → benchmark → recalculate weights
# --------------------------------------------------------------------------- #
def _make_session():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _add_prediction(session, entity, predicted, agent_scores) -> Prediction:
    pred = Prediction(
        entity=entity, domain="finance", prediction=predicted,
        score=0.5, confidence=0.5, contributors=[],
    )
    session.add(pred)
    session.flush()
    for agent, score in agent_scores.items():
        session.add(
            AgentResult(
                prediction_id=pred.id, agent_name=agent, score=score,
                confidence=0.7, weight=0.2,
            )
        )
    session.flush()
    return pred


def test_record_outcome_sets_correct_flag() -> None:
    session = _make_session()
    try:
        svc = PredictionService(session)
        pred = _add_prediction(session, "Acme", "Positive", {"historical": 0.8})
        out = svc.record_outcome(pred.id, "Positive")
        assert out is not None
        assert out.correct is True
        # Updating the same prediction's outcome overwrites, not duplicates.
        out2 = svc.record_outcome(pred.id, "Negative")
        assert out2.correct is False
    finally:
        session.close()


def test_record_outcome_unknown_prediction_returns_none() -> None:
    import uuid

    session = _make_session()
    try:
        assert PredictionService(session).record_outcome(uuid.uuid4(), "Positive") is None
    finally:
        session.close()


def test_recalculate_weights_adjusts_and_snapshots() -> None:
    session = _make_session()
    wm = WeightManager()
    wm._cache.delete(_CACHE_KEY)
    try:
        svc = PredictionService(session, weight_manager=wm)
        # 'historical' always calls the realised positive class right; 'risk' always wrong.
        for i in range(6):
            pred = _add_prediction(
                session, f"E{i}", "Positive", {"historical": 0.9, "risk": 0.1}
            )
            svc.record_outcome(pred.id, "Positive")  # label = 1
        session.commit()

        accuracy = svc.agent_accuracy()
        assert accuracy["historical"] == 1.0
        assert accuracy["risk"] == 0.0

        adjusted, samples, acc, weights = svc.recalculate_weights()
        assert adjusted is True
        assert samples == 12  # 6 predictions x 2 agents
        # The accurate agent ends weighted at least as high as the inaccurate one.
        assert weights["historical"] > weights["risk"]
        # Snapshots persisted: one per agent + a global row.
        snaps = list(session.scalars(select(AccuracySnapshot)))
        names = {s.agent_name for s in snaps}
        assert "historical" in names and "__global__" in names
    finally:
        wm._cache.delete(_CACHE_KEY)
        session.close()


def test_recalculate_with_no_outcomes_is_noop() -> None:
    session = _make_session()
    wm = WeightManager()
    wm._cache.delete(_CACHE_KEY)
    try:
        svc = PredictionService(session, weight_manager=wm)
        adjusted, samples, acc, weights = svc.recalculate_weights()
        assert adjusted is False
        assert samples == 0
        assert acc == {}
        assert weights == wm.current()
    finally:
        wm._cache.delete(_CACHE_KEY)
        session.close()
