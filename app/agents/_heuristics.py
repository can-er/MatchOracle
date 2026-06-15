"""Deterministic heuristic helpers shared by the stub agents (Sprint 03).

Real data sources are wired in Phase 2 (connectors/MCP). Until then agents
derive **reproducible** scores from the entity so tests are deterministic and
the orchestration flow is exercised realistically.
"""

from __future__ import annotations

from app.agents.base import AgentContext


def derived_score(ctx: AgentContext, salt: str, *, low: float = 0.2, high: float = 0.9) -> float:
    """Map the context seed into a stable score in ``[low, high]``."""
    seed = ctx.seed(salt)
    unit = (seed % 10_000) / 10_000.0
    return round(low + unit * (high - low), 3)


def derived_confidence(
    ctx: AgentContext, salt: str, *, low: float = 0.5, high: float = 0.9
) -> float:
    seed = ctx.seed(salt + "::conf")
    unit = (seed % 10_000) / 10_000.0
    return round(low + unit * (high - low), 3)


def signal_from_context(ctx: AgentContext, keys: list[str]) -> float | None:
    """Pull a numeric nudge from ``ctx.context`` if the caller supplied hints."""
    values: list[float] = []
    for key in keys:
        value = ctx.context.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    avg = sum(values) / len(values)
    # Clamp to [0,1]; callers may pass 0-1 or 0-100.
    if avg > 1:
        avg = avg / 100.0
    return max(0.0, min(1.0, avg))
