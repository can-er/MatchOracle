"""Feedback loop, closed-loop self-improvement & multi-model router (Sprint 12)."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import models  # noqa: F401 — register models
from app.db.base import Base
from app.db.models import AgentResult, Prediction
from app.llm.router import ModelRouter, ModelTier
from app.orchestration.service import PredictionService
from app.orchestration.weights import _CACHE_KEY, WeightManager
from app.prediction.feedback import feedback_reward, label_from_feedback


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_feedback_reward() -> None:
    assert feedback_reward("approve") == 1.0
    assert feedback_reward("reject") == -1.0
    assert feedback_reward("correct", "Positive") == 0.5
    assert feedback_reward("correct", None) == 0.0
    assert feedback_reward("whatever") == 0.0


def test_label_from_feedback() -> None:
    # approve keeps the predicted class; reject flips it; correct reads the fix.
    assert label_from_feedback("approve", "Positive") == 1
    assert label_from_feedback("reject", "Positive") == 0
    assert label_from_feedback("reject", "Negative") == 1
    assert label_from_feedback("correct", "Positive", "Negative") == 0
    # A non-binary predicted label can't be flipped on reject.
    assert label_from_feedback("reject", "2-1") is None


# --------------------------------------------------------------------------- #
# Multi-model router policies (story 12-4)
# --------------------------------------------------------------------------- #
def _router(policy: str) -> ModelRouter:
    tiers = [
        ModelTier("small", "m-small", cost=0.2, quality=0.7),
        ModelTier("large", "m-large", cost=1.0, quality=0.95),
    ]
    return ModelRouter(policy=policy, tiers=tiers)


def test_router_cost_policy_picks_cheapest() -> None:
    assert _router("cost").select(complexity=0.9).name == "small"


def test_router_quality_policy_picks_best() -> None:
    assert _router("quality").select(complexity=0.1).name == "large"


def test_router_balanced_escalates_with_complexity() -> None:
    balanced = _router("balanced")
    assert balanced.select(complexity=0.2).name == "small"
    assert balanced.select(complexity=0.8).name == "large"


# --------------------------------------------------------------------------- #
# Closed loop: a human correction reshapes future weights (Sprint 12 DoD)
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


def test_record_feedback_sets_reward() -> None:
    session = _make_session()
    try:
        svc = PredictionService(session)
        pred = _add_prediction(session, "Acme", "Positive", {"historical": 0.8})
        fb = svc.record_feedback(pred.id, "approve", validator="alice")
        assert fb is not None
        assert fb.reward == 1.0
        assert fb.validator == "alice"
    finally:
        session.close()


def test_learn_from_feedback_closes_the_loop() -> None:
    session = _make_session()
    wm = WeightManager()
    wm._cache.delete(_CACHE_KEY)
    try:
        svc = PredictionService(session, weight_manager=wm)
        before = wm.current()
        # 'historical' agrees with the human-confirmed positive class; 'risk' opposes it.
        for i in range(6):
            pred = _add_prediction(
                session, f"E{i}", "Positive", {"historical": 0.9, "risk": 0.1}
            )
            svc.record_feedback(pred.id, "approve")  # label = 1
        session.commit()

        adjusted, samples, accuracy, weights = svc.learn_from_feedback()
        assert adjusted is True
        assert samples == 12
        assert accuracy["historical"] == 1.0 and accuracy["risk"] == 0.0
        # Future behaviour changed: the agreeing agent is now weighted higher.
        assert weights["historical"] > before["historical"]
        assert weights["historical"] > weights["risk"]

        # Guardrail: rollback restores the configured defaults.
        restored = svc.rollback_weights()
        assert restored == before
    finally:
        wm._cache.delete(_CACHE_KEY)
        session.close()


def test_learn_from_feedback_noop_without_feedback() -> None:
    session = _make_session()
    wm = WeightManager()
    wm._cache.delete(_CACHE_KEY)
    try:
        adjusted, samples, accuracy, weights = PredictionService(
            session, weight_manager=wm
        ).learn_from_feedback()
        assert adjusted is False and samples == 0 and accuracy == {}
    finally:
        wm._cache.delete(_CACHE_KEY)
        session.close()
