"""Aggregate API v1 router (Sprint 06).

Mounts every versioned route module under ``settings.api_prefix`` (``/api/v1``).
The unversioned ``/health`` probe is wired directly in ``app.main``.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import (
    accuracy,
    auth,
    connectors,
    feedback,
    mcp,
    predictions,
    worldcup,
)
from app.config import settings

api_router = APIRouter(prefix=settings.api_prefix)
api_router.include_router(predictions.router)
api_router.include_router(connectors.router)
api_router.include_router(worldcup.router)
api_router.include_router(mcp.router)
api_router.include_router(accuracy.router)
api_router.include_router(feedback.router)
api_router.include_router(auth.router)
