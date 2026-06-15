"""Accuracy, benchmarking & auto-weighting endpoints (Sprints 10 & 11).

Record real-world outcomes, inspect per-agent accuracy/benchmark, and trigger an
accuracy-driven weight recalculation. The weighting itself lives in the
:class:`~app.orchestration.weights.WeightManager` (guardrails: clamp + smoothing).
"""

from __future__ import annotations

import uuid
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.base import get_session
from app.orchestration.service import get_prediction_service
from app.orchestration.weights import get_weight_manager
from app.schemas.benchmark import (
    AgentBenchmarkRead,
    AutoTuneResponse,
    BenchmarkReport,
    OutcomeCreate,
    OutcomeRead,
    RecalculateResponse,
    WeightsResponse,
)
from app.security.principal import Principal, require_role

router = APIRouter(tags=["accuracy"])


@router.post(
    "/predictions/{prediction_id}/outcome",
    response_model=OutcomeRead,
    status_code=status.HTTP_201_CREATED,
    summary="Record the real-world outcome of a prediction",
)
def record_outcome(
    prediction_id: uuid.UUID,
    payload: OutcomeCreate,
    session: Session = Depends(get_session),
    principal: Principal = Depends(require_role("analyst")),
) -> OutcomeRead:
    outcome = get_prediction_service(session).record_outcome(
        prediction_id,
        payload.actual,
        actual_score=payload.actual_score,
        notes=payload.notes,
    )
    if outcome is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Prediction not found")
    return OutcomeRead(
        prediction_id=str(outcome.prediction_id),
        actual=outcome.actual,
        actual_score=outcome.actual_score,
        correct=outcome.correct,
        notes=outcome.notes,
    )


@router.get(
    "/agents/accuracy",
    response_model=dict[str, float],
    summary="Per-agent directional accuracy over resolved predictions",
)
def agent_accuracy(session: Session = Depends(get_session)) -> dict[str, float]:
    return get_prediction_service(session).agent_accuracy()


@router.get(
    "/agents/benchmark",
    response_model=BenchmarkReport,
    summary="Per-agent benchmark report (accuracy, calibration, contribution)",
)
def agent_benchmark(session: Session = Depends(get_session)) -> BenchmarkReport:
    evaluated, report = get_prediction_service(session).benchmark()
    return BenchmarkReport(
        evaluated=evaluated,
        agents=[AgentBenchmarkRead(**asdict(b)) for b in report],
    )


@router.get("/weights", response_model=WeightsResponse, summary="Current agent weights")
def current_weights() -> WeightsResponse:
    return WeightsResponse(weights=get_weight_manager().current())


@router.post(
    "/weights/recalculate",
    response_model=RecalculateResponse,
    summary="Recalculate agent weights from measured accuracy (guardrailed)",
)
def recalculate_weights(
    session: Session = Depends(get_session),
    principal: Principal = Depends(require_role("admin")),
) -> RecalculateResponse:
    adjusted, samples, accuracy, weights = get_prediction_service(session).recalculate_weights()
    return RecalculateResponse(
        adjusted=adjusted, samples=samples, accuracy=accuracy, weights=weights
    )


@router.post(
    "/weights/auto-tune",
    response_model=AutoTuneResponse,
    summary="Autonomous loop: ingest real World Cup results → re-tune weights (no human)",
)
def auto_tune(
    session: Session = Depends(get_session),
    principal: Principal = Depends(require_role("admin")),
) -> AutoTuneResponse:
    return AutoTuneResponse(**get_prediction_service(session).autonomous_learn())
