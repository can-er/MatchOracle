"""Distributed execution (Sprint 14).

A Celery app + tasks let agent work fan out to workers over a Redis broker when
``settings.distributed_agents`` is on. When it's off (default), Celery runs in
eager mode — tasks execute in-process with no broker — so the path is fully
exercisable in tests and local runs.
"""

from __future__ import annotations

from app.workers.celery_app import celery_app
from app.workers.tasks import run_agent_task, run_agents_distributed

__all__ = ["celery_app", "run_agent_task", "run_agents_distributed"]
