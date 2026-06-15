"""Health & readiness endpoints (Sprint 00).

``/health`` is a cheap liveness probe used by Docker/Kubernetes and the Sprint 00
Definition of Done. It must not depend on Postgres/Redis being up.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__
from app.config import settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health() -> HealthResponse:
    """Return a static OK payload — proves the app process is up."""
    return HealthResponse(
        status="ok",
        service="matchoracle",
        version=__version__,
        environment=settings.env,
    )
