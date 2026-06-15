"""Celery application (Sprint 14).

Broker/result backend come from settings (Redis DBs 1 & 2). With
``distributed_agents`` off, ``task_always_eager`` makes tasks run locally and
synchronously, so no worker/broker is needed for tests or single-node runs.
"""

from __future__ import annotations

from celery import Celery

from app.config import settings

celery_app = Celery(
    "matchoracle",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_always_eager=not settings.distributed_agents,
    task_eager_propagates=True,
    broker_connection_retry_on_startup=True,
)
