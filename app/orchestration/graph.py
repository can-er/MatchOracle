"""LangGraph orchestration graph (Sprints 02 & 05).

Builds a ``StateGraph`` that fans out to every analytical agent in parallel,
then runs the Expert agent last (it consumes the peers' outputs). The graph is
generic: agents are pulled from the registry, so adding an agent needs no graph
change (Sprint 02 DoD).
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.agents.base import AgentContext
from app.agents.registry import registry
from app.logging_config import get_logger
from app.orchestration.state import GraphState

logger = get_logger(__name__)

# Agents that run in the parallel fan-out (everyone except the expert).
EXPERT_AGENT = "expert"


def _make_agent_node(agent_name: str):
    """Return a node callable that runs one agent and appends its result."""

    def _node(state: GraphState) -> dict:
        agent = registry.create(agent_name)
        ctx = AgentContext(
            entity=state["entity"],
            domain=state.get("domain"),
            context=dict(state.get("context", {})),
            connectors=state.get("context", {}).get("_connectors", []),
            mcp=state.get("context", {}).get("_mcp"),
        )
        result = agent.run(ctx)
        return {"results": [result]}

    return _node


def _expert_node(state: GraphState) -> dict:
    agent = registry.create(EXPERT_AGENT)
    peer_results = [r.model_dump() for r in state.get("results", [])]
    ctx_extra = dict(state.get("context", {}))
    ctx_extra["peer_results"] = peer_results
    ctx = AgentContext(
        entity=state["entity"],
        domain=state.get("domain"),
        context=ctx_extra,
    )
    result = agent.run(ctx)
    return {"results": [result]}


@lru_cache
def build_graph():
    """Compile and cache the orchestration graph."""
    builder = StateGraph(GraphState)

    analytical = [n for n in registry.names() if n not in (EXPERT_AGENT, "mock")]

    for name in analytical:
        builder.add_node(name, _make_agent_node(name))
        builder.add_edge(START, name)

    has_expert = registry.get(EXPERT_AGENT) is not None
    if has_expert:
        builder.add_node(EXPERT_AGENT, _expert_node)
        for name in analytical:
            builder.add_edge(name, EXPERT_AGENT)
        builder.add_edge(EXPERT_AGENT, END)
    else:  # pragma: no cover - expert is always registered in practice
        for name in analytical:
            builder.add_edge(name, END)

    compiled = builder.compile()
    logger.info("orchestration.graph.compiled", agents=analytical, expert=has_expert)
    return compiled
