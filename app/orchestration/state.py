"""LangGraph state definition (Sprint 02)."""

from __future__ import annotations

from typing import Annotated, TypedDict

from app.schemas.agent import AgentResult


def _merge_results(left: list[AgentResult], right: list[AgentResult]) -> list[AgentResult]:
    """Reducer so parallel agent nodes can append to the same list."""
    return [*left, *right]


class GraphState(TypedDict, total=False):
    """Shared state flowing through the orchestration graph."""

    entity: str
    domain: str | None
    context: dict
    # Annotated reducer lets fan-out agent nodes append concurrently.
    results: Annotated[list[AgentResult], _merge_results]
    expert_result: list[AgentResult]
