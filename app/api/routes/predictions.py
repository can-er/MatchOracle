"""Prediction endpoints (Sprint 06).

Exposes the explainable prediction surface over the end-to-end
:class:`~app.orchestration.service.PredictionService`.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.base import get_session
from app.orchestration.service import get_prediction_service
from app.schemas.agent import AgentResultRead
from app.schemas.common import Page
from app.schemas.prediction import PredictionDetail, PredictionRequest, PredictionResponse
from app.security.audit import record_audit, tenant_scope
from app.security.principal import Principal, current_principal, require_role

router = APIRouter(tags=["predictions"])


def _visible(prediction, principal: Principal) -> bool:
    """Tenant isolation: a principal only sees its own tenant's data (when auth on)."""
    scope = tenant_scope(principal)
    return scope is None or prediction.tenant_id == scope


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate an explainable prediction for an entity",
)
def create_prediction(
    payload: PredictionRequest,
    session: Session = Depends(get_session),
    principal: Principal = Depends(require_role("analyst")),
) -> PredictionResponse:
    tenant = tenant_scope(principal)
    prediction = get_prediction_service(session).predict(payload, tenant_id=tenant)
    record_audit(
        session, principal.username, "prediction.create",
        resource=payload.entity, tenant_id=tenant,
    )
    return PredictionResponse.model_validate(prediction)


@router.get(
    "/predictions",
    response_model=Page[PredictionResponse],
    summary="List past predictions (paginated)",
)
def list_predictions(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
    principal: Principal = Depends(current_principal),
) -> Page[PredictionResponse]:
    items, total = get_prediction_service(session).list_predictions(
        limit=limit, offset=offset, tenant_id=tenant_scope(principal)
    )
    return Page[PredictionResponse](
        items=[PredictionResponse.model_validate(p) for p in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionDetail,
    summary="Prediction detail with per-agent contributions",
)
def get_prediction(
    prediction_id: uuid.UUID,
    session: Session = Depends(get_session),
    principal: Principal = Depends(current_principal),
) -> PredictionDetail:
    prediction = get_prediction_service(session).get(prediction_id)
    if prediction is None or not _visible(prediction, principal):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Prediction not found")
    return PredictionDetail.model_validate(prediction)


@router.get(
    "/predictions/{prediction_id}/agents",
    response_model=list[AgentResultRead],
    summary="Per-agent results behind a prediction",
)
def get_prediction_agents(
    prediction_id: uuid.UUID,
    session: Session = Depends(get_session),
    principal: Principal = Depends(current_principal),
) -> list[AgentResultRead]:
    prediction = get_prediction_service(session).get(prediction_id)
    if prediction is None or not _visible(prediction, principal):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Prediction not found")
    return [AgentResultRead.model_validate(a) for a in prediction.agent_results]
