"""Connector management endpoints (Sprint 06).

CRUD surface only — the actual connector execution logic lands in Sprint 08.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.base import get_session
from app.db.models import Connector
from app.repositories.connector_repository import ConnectorRepository
from app.schemas.connector import ConnectorCreate, ConnectorRead
from app.security.principal import Principal, require_role

router = APIRouter(tags=["connectors"])


@router.get("/connectors", response_model=list[ConnectorRead], summary="List connectors")
def list_connectors(session: Session = Depends(get_session)) -> list[ConnectorRead]:
    connectors = ConnectorRepository(session).list_all()
    return [ConnectorRead.model_validate(c) for c in connectors]


@router.post(
    "/connectors",
    response_model=ConnectorRead,
    status_code=status.HTTP_201_CREATED,
    summary="Register a connector",
)
def create_connector(
    payload: ConnectorCreate,
    session: Session = Depends(get_session),
    principal: Principal = Depends(require_role("admin")),
) -> ConnectorRead:
    repo = ConnectorRepository(session)
    if repo.get_by_name(payload.name) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Connector '{payload.name}' already exists")
    connector = Connector(
        name=payload.name,
        type=payload.type.value,
        configuration=payload.configuration,
        status="inactive",
    )
    repo.add(connector)
    return ConnectorRead.model_validate(connector)
