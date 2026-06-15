"""Sprint 02 orchestration DoD tests.

Covers the LangGraph orchestration graph (build + invoke), the GraphState
parallel-merge reducer, and the "add an agent = one class + one registration,
zero graph change" contract on a *fresh* registry (never the global one).
"""

from __future__ import annotations

import typing

import pytest

# Importing app.agents registers every built-in agent on the global registry,
# which build_graph() reads from. Keep this import for the side effect.
import app.agents  # noqa: F401
from app.agents.base import AgentContext, BaseAgent
from app.agents.registry import AgentRegistry
from app.orchestration.graph import build_graph
from app.orchestration.state import GraphState, _merge_results
from app.schemas.agent import AgentResult

# The six analytical + expert agents the orchestration must always cover.
EXPECTED_AGENTS = {"historical", "trend", "contextual", "risk", "market", "expert"}


# --------------------------------------------------------------------------- #
# build_graph() compiles and invoke() returns all analytical + expert results
# --------------------------------------------------------------------------- #
def test_build_graph_compiles_and_is_cached():
    graph = build_graph()
    assert graph is not None
    # lru_cache means the compiled graph is reused across calls.
    assert build_graph() is graph
    # A compiled LangGraph exposes invoke().
    assert hasattr(graph, "invoke")


def test_invoke_returns_results_list_of_agent_results():
    graph = build_graph()
    out = graph.invoke({"entity": "Acme", "domain": "sports", "context": {}})

    assert isinstance(out, dict)
    assert "results" in out
    results = out["results"]
    assert isinstance(results, list)
    assert all(isinstance(r, AgentResult) for r in results)


def test_invoke_covers_all_six_agents():
    graph = build_graph()
    out = graph.invoke({"entity": "Acme", "domain": "sports", "context": {}})
    results = out["results"]

    produced = {r.agent for r in results}
    # Every expected analytical + expert agent must have produced a result.
    assert EXPECTED_AGENTS.issubset(produced), f"missing agents: {EXPECTED_AGENTS - produced}"
    # At least the six required agents ran (mock is excluded by the graph).
    assert len(results) >= 6


def test_invoke_results_satisfy_contract():
    graph = build_graph()
    out = graph.invoke({"entity": "Acme", "domain": "sports", "context": {}})

    for r in out["results"]:
        assert isinstance(r.agent, str) and r.agent
        assert isinstance(r.score, float)
        assert 0.0 <= r.score <= 1.0
        assert isinstance(r.confidence, float)
        assert 0.0 <= r.confidence <= 1.0


def test_invoke_is_deterministic_across_runs():
    graph = build_graph()
    payload = {"entity": "Acme", "domain": "sports", "context": {}}

    out1 = graph.invoke(payload)
    out2 = graph.invoke(payload)

    by_agent_1 = {r.agent: (r.score, r.confidence) for r in out1["results"]}
    by_agent_2 = {r.agent: (r.score, r.confidence) for r in out2["results"]}

    # Heuristic agents derive scores from a deterministic entity hash seed,
    # so two identical invocations must produce identical per-agent scores.
    for name in EXPECTED_AGENTS:
        assert by_agent_1[name] == by_agent_2[name]


def test_invoke_mock_agent_is_excluded_from_graph():
    graph = build_graph()
    out = graph.invoke({"entity": "Acme", "domain": "sports", "context": {}})
    produced = {r.agent for r in out["results"]}
    # The graph explicitly excludes the "mock" agent from the fan-out.
    assert "mock" not in produced


# --------------------------------------------------------------------------- #
# GraphState reducer merges parallel agent outputs
# --------------------------------------------------------------------------- #
def test_merge_results_concatenates_parallel_outputs():
    a = AgentResult(agent="historical", score=0.4, confidence=0.5)
    b = AgentResult(agent="trend", score=0.6, confidence=0.7)

    merged = _merge_results([a], [b])
    assert merged == [a, b]
    assert len(merged) == 2


def test_merge_results_handles_empty_sides():
    a = AgentResult(agent="risk", score=0.2, confidence=0.3)
    assert _merge_results([], [a]) == [a]
    assert _merge_results([a], []) == [a]
    assert _merge_results([], []) == []


def test_merge_results_is_annotated_reducer_on_state():
    # The reducer is wired into GraphState so fan-out nodes append concurrently.
    # The state module uses `from __future__ import annotations`, so resolve the
    # string annotations with include_extras to recover the Annotated metadata.
    hints = typing.get_type_hints(GraphState, include_extras=True)
    assert "results" in hints
    metadata = getattr(hints["results"], "__metadata__", ())
    assert _merge_results in metadata


# --------------------------------------------------------------------------- #
# "Add an agent = one class + one registration, zero graph change"
# --------------------------------------------------------------------------- #
def test_register_new_agent_on_fresh_registry():
    # Use a FRESH registry so we never pollute the global one.
    fresh = AgentRegistry()

    @fresh.register
    class DummyAgent(BaseAgent):
        name = "dummy_sprint02"
        description = "test-only agent"

        def analyze(self, ctx: AgentContext) -> AgentResult:
            return AgentResult(agent=self.name, score=0.5, confidence=0.5)

    # One class + one registration is enough: it is discoverable and creatable.
    assert "dummy_sprint02" in fresh.names()
    assert fresh.get("dummy_sprint02") is DummyAgent

    instance = fresh.create("dummy_sprint02")
    assert isinstance(instance, DummyAgent)
    assert instance.name == "dummy_sprint02"

    # And it actually runs through the safe BaseAgent.run wrapper.
    result = instance.run(AgentContext(entity="Acme"))
    assert isinstance(result, AgentResult)
    assert result.agent == "dummy_sprint02"


def test_register_rejects_base_name():
    fresh = AgentRegistry()

    class BaseNamed(BaseAgent):
        name = "base"

    with pytest.raises(ValueError):
        fresh.register(BaseNamed)


def test_register_rejects_missing_name():
    fresh = AgentRegistry()

    class NoName(BaseAgent):
        name = ""  # falsy -> rejected

    with pytest.raises(ValueError):
        fresh.register(NoName)


def test_fresh_registry_does_not_affect_global():
    # A class registered on a fresh registry must not leak into the global one.
    fresh = AgentRegistry()

    @fresh.register
    class Isolated(BaseAgent):
        name = "isolated_sprint02"

        def analyze(self, ctx: AgentContext) -> AgentResult:
            return AgentResult(agent=self.name, score=0.5, confidence=0.5)

    from app.agents.registry import registry as global_registry

    assert "isolated_sprint02" not in global_registry.names()
