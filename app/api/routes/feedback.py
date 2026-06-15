"""Human-in-the-loop feedback, closed-loop learning & multi-model routing (Sprint 12).

Endpoints to inject a human verdict, close the self-improvement loop (verdicts
reshape agent weights), roll the weights back (guardrail), and inspect the
multi-model router's decision under the active policy.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.base import get_session
from app.llm.router import get_model_router
from app.orchestration.service import get_prediction_service
from app.schemas.benchmark import WeightsResponse
from app.schemas.feedback import (
    FeedbackCreate,
    FeedbackRead,
    LearnResponse,
    ModelRouteResponse,
)
from app.security.principal import Principal, require_role

router = APIRouter(tags=["feedback"])


@router.post(
    "/predictions/{prediction_id}/feedback",
    response_model=FeedbackRead,
    status_code=status.HTTP_201_CREATED,
    summary="Record a human validation/correction (approve | reject | correct)",
)
def record_feedback(
    prediction_id: uuid.UUID,
    payload: FeedbackCreate,
    session: Session = Depends(get_session),
    principal: Principal = Depends(require_role("analyst")),
) -> FeedbackRead:
    feedback = get_prediction_service(session).record_feedback(
        prediction_id,
        payload.verdict,
        validator=payload.validator,
        corrected_prediction=payload.corrected_prediction,
        comment=payload.comment,
    )
    if feedback is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Prediction not found")
    return FeedbackRead(
        id=str(feedback.id),
        prediction_id=str(feedback.prediction_id),
        verdict=feedback.verdict,
        validator=feedback.validator,
        corrected_prediction=feedback.corrected_prediction,
        reward=feedback.reward,
        comment=feedback.comment,
    )


@router.post(
    "/weights/learn-from-feedback",
    response_model=LearnResponse,
    summary="Close the loop: human verdicts reshape agent weights (guardrailed)",
)
def learn_from_feedback(
    session: Session = Depends(get_session),
    principal: Principal = Depends(require_role("admin")),
) -> LearnResponse:
    adjusted, samples, accuracy, weights = get_prediction_service(session).learn_from_feedback()
    return LearnResponse(adjusted=adjusted, samples=samples, accuracy=accuracy, weights=weights)


@router.post(
    "/weights/rollback",
    response_model=WeightsResponse,
    summary="Guardrail: roll agent weights back to the configured defaults",
)
def rollback_weights(
    session: Session = Depends(get_session),
    principal: Principal = Depends(require_role("admin")),
) -> WeightsResponse:
    return WeightsResponse(weights=get_prediction_service(session).rollback_weights())


@router.get(
    "/llm/router",
    response_model=ModelRouteResponse,
    summary="Inspect the multi-model router's choice under the active policy",
)
def inspect_router(complexity: float = Query(0.5, ge=0.0, le=1.0)) -> ModelRouteResponse:
    router_ = get_model_router()
    tier = router_.select(complexity=complexity)
    return ModelRouteResponse(
        policy=router_.policy,
        complexity=complexity,
        selected_tier=tier.name,
        selected_model=router_._effective_model(tier),
        tiers=[asdict(t) for t in router_.tiers],
    )
