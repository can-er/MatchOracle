"""Distributed agent execution via Celery in eager mode (Sprint 14).

No broker required: ``distributed_agents`` is off by default, so Celery runs the
tasks in-process. This proves the distributed code path end-to-end offline.
"""

from __future__ import annotations

from app.workers.tasks import run_agent_task, run_agents_distributed


def test_run_agent_task_returns_serialisable_result() -> None:
    result = run_agent_task("historical", "Acme", "sports")
    assert result["agent"] == "historical"
    assert 0.0 <= result["score"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["extra"], dict)  # JSON-safe payload


def test_run_agents_distributed_gathers_all_analytical_agents() -> None:
    results = run_agents_distributed("Acme", "sports")
    names = {r["agent"] for r in results}
    assert {"historical", "trend", "contextual", "risk", "market"} <= names
    # The expert is excluded from the parallel fan-out.
    assert "expert" not in names
    assert all(0.0 <= r["score"] <= 1.0 for r in results)


def test_distributed_is_deterministic_for_same_entity() -> None:
    a = {r["agent"]: r["score"] for r in run_agents_distributed("Same Co", "sports")}
    b = {r["agent"]: r["score"] for r in run_agents_distributed("Same Co", "sports")}
    assert a == b
