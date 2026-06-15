"""FastAPI application entrypoint (Sprints 00 & 06).

Exposes the ``app`` ASGI object used by uvicorn (``uvicorn app.main:app``):
- ``/health`` liveness probe (Sprint 00)
- the versioned REST surface under ``settings.api_prefix`` (Sprint 06)

The database schema is managed by Alembic migrations (``alembic upgrade head``),
applied as a separate step — not at app startup.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from app import __version__
from app.api.router import api_router
from app.api.routes import health
from app.config import settings
from app.logging_config import configure_logging, get_logger
from app.observability.metrics import CONTENT_TYPE, observe_request, render_metrics

logger = get_logger(__name__)

_DASHBOARD = Path(__file__).parent / "web" / "dashboard.html"


def _route_template(request: Request) -> str:
    """Low-cardinality path label: the matched route template, else the raw path."""
    route = request.scope.get("route")
    return getattr(route, "path", None) or request.url.path


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown hooks.

    The schema is owned by Alembic migrations (``alembic upgrade head``), run as a
    separate step (container entrypoint / CI / local) — never at app startup.
    """
    logger.info("app.startup", database=settings.database_url.split("://")[0])
    yield


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    configure_logging()

    app = FastAPI(
        title="MatchOracle",
        description="AI-Powered Multi-Agent Prediction Platform",
        version=__version__,
        lifespan=lifespan,
    )

    app.include_router(health.router)  # /health (unversioned, Sprint 00)
    app.include_router(api_router)  # /api/v1/* (Sprint 06)

    if settings.metrics_enabled:

        @app.middleware("http")
        async def _instrument(request: Request, call_next):  # Sprint 14
            start = time.perf_counter()
            response = await call_next(request)
            observe_request(
                request.method,
                _route_template(request),
                response.status_code,
                time.perf_counter() - start,
            )
            return response

        @app.get("/metrics", include_in_schema=False)
        def metrics() -> Response:
            """Prometheus exposition (Sprint 14)."""
            return Response(content=render_metrics(), media_type=CONTENT_TYPE)

    @app.get("/api/cron/tick", include_in_schema=False)
    def cron_tick(request: Request) -> dict:
        """Vercel Cron entrypoint (daily). Phase 1: authenticated stub.

        Phase 5 fills this in: refresh World Cup predictions → autonomous_learn →
        find the next open MPP game week → predict full-agents → submit forecasts.
        Vercel sends ``Authorization: Bearer $CRON_SECRET`` when CRON_SECRET is set.
        """
        import os

        secret = os.environ.get("CRON_SECRET")
        if secret and request.headers.get("authorization") != f"Bearer {secret}":
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        logger.info("cron.tick.stub")
        return {"status": "stub", "todo": "phase 5: refresh + autonomous_learn + MPP submit"}

    @app.get("/", include_in_schema=False)
    @app.get("/dashboard", include_in_schema=False)
    def dashboard() -> FileResponse:
        """Serve the World Cup predictions dashboard (Sprint 09)."""
        return FileResponse(_DASHBOARD, media_type="text/html")

    logger.info("app.created", env=settings.env, api_prefix=settings.api_prefix)
    return app


app = create_app()
