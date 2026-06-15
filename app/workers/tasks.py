"""Celery tasks: distributed agent execution (Sprint 14, story 14-1).

``run_agent_task`` is the unit of distributed work — one agent, one entity.
``run_agents_distributed`` fans the analytical agents out across workers and
collects their results, mirroring the in-process graph's parallel stage. The
expert agent stays in-process (it consumes the peers' outputs).
"""

from __future__ import annotations

from app.config import settings
from app.logging_config import get_logger
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(name="agents.run")
def run_agent_task(
    agent_name: str, entity: str, domain: str | None = None, context: dict | None = None
) -> dict:
    """Run a single agent and return its result as a plain dict (JSON-safe)."""
    from app.agents.base import AgentContext
    from app.agents.registry import registry

    agent = registry.create(agent_name)
    ctx = AgentContext(entity=entity, domain=domain, context=dict(context or {}))
    return agent.run(ctx).model_dump()


def run_agents_distributed(
    entity: str, domain: str | None = None, context: dict | None = None
) -> list[dict]:
    """Dispatch the analytical agents as Celery tasks and gather their results.

    Honours ``distributed_agents``: on → real ``apply_async`` over the broker;
    off → eager in-process execution (no broker needed). The expert is excluded
    (it runs last, in-process, over the peers).
    """
    from app.agents.registry import registry

    analytical = [n for n in registry.names() if n not in ("expert", "mock")]
    if settings.distributed_agents:
        async_results = [
            run_agent_task.apply_async(args=(name, entity, domain, context))
            for name in analytical
        ]
        results = [ar.get(timeout=settings.llm_timeout_seconds + 30) for ar in async_results]
    else:
        # Eager / single-node: run the task body locally, no broker round-trip.
        results = [
            run_agent_task.apply(args=(name, entity, domain, context)).get()
            for name in analytical
        ]
    logger.info(
        "agents.distributed.gathered",
        count=len(results),
        distributed=settings.distributed_agents,
    )
    return results
