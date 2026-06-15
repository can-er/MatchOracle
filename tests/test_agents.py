"""Agent contract & behaviour tests (closes Sprint 03 & 04 agent DoD).

Covers the registered analytical agents (heuristic + expert + mock):

* every agent returns a valid, correctly-named ``AgentResult`` with
  ``score``/``confidence`` in ``[0, 1]``;
* the heuristic agents are deterministic for the same entity;
* the risk agent exposes a ``risk_level`` in ``extra``;
* the expert agent stays valid (with a synthesised explanation) when no LLM
  is reachable — the case in this offline environment;
* ``BaseAgent.run`` is a safety net: a failing ``analyze`` yields a neutral
  fallback and never raises;
* ``AgentResult`` rejects out-of-range ``score``/``confidence``.

No network/LLM is available, so assertions target contracts and behaviour
(ranges, types, determinism, names) — never magic numbers or live LLM text.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

# Importing the package registers every built-in agent via the registry.
from app.agents import registry
from app.agents.base import AgentContext, BaseAgent
from app.schemas.agent import AgentResult

# The heuristic stub agents (Sprint 03) derive a stable score from the entity.
HEURISTIC_AGENTS = ["historical", "trend", "contextual", "risk", "market"]
# All registered agents, including the expert (LLM) and mock.
ALL_AGENTS = HEURISTIC_AGENTS + ["expert", "mock"]


def _ctx(entity: str = "Acme", domain: str | None = "sports", **kw) -> AgentContext:
    return AgentContext(entity=entity, domain=domain, **kw)


# --------------------------------------------------------------------------- #
# Registry wiring
# --------------------------------------------------------------------------- #
def test_all_expected_agents_are_registered() -> None:
    names = set(registry.names())
    for name in ALL_AGENTS:
        assert name in names, f"agent {name!r} not registered"


# --------------------------------------------------------------------------- #
# Contract: every agent yields a valid, correctly-named AgentResult
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ALL_AGENTS)
def test_agent_run_returns_valid_result(name: str) -> None:
    agent = registry.create(name)
    result = agent.run(_ctx())

    assert isinstance(result, AgentResult)
    assert result.agent == name
    assert isinstance(result.score, float)
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.extra, dict)


@pytest.mark.parametrize("name", ALL_AGENTS)
def test_agent_run_never_raises(name: str) -> None:
    """The orchestrator relies on run() being exception-safe for any agent."""
    # Empty context (no hints, no mcp) must still produce a result.
    result = registry.create(name).run(AgentContext(entity="Anything"))
    assert isinstance(result, AgentResult)


# --------------------------------------------------------------------------- #
# Determinism of the heuristic agents
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", HEURISTIC_AGENTS)
def test_heuristic_agent_is_deterministic_for_same_entity(name: str) -> None:
    """Two fresh instances/contexts for the same entity must agree."""
    r1 = registry.create(name).run(_ctx(entity="Same Entity"))
    r2 = registry.create(name).run(_ctx(entity="Same Entity"))
    assert r1.score == r2.score
    assert r1.confidence == r2.confidence


@pytest.mark.parametrize("name", HEURISTIC_AGENTS)
def test_heuristic_agent_varies_with_entity(name: str) -> None:
    """Different entities should generally produce different seeds/scores.

    Asserts behaviour (entity-sensitivity) without pinning any value.
    """
    a = registry.create(name).run(_ctx(entity="Entity Alpha"))
    b = registry.create(name).run(_ctx(entity="Totally Different Entity"))
    # Scores are still each within range regardless of equality.
    assert 0.0 <= a.score <= 1.0
    assert 0.0 <= b.score <= 1.0
    assert a.score != b.score


# --------------------------------------------------------------------------- #
# Risk agent specifics
# --------------------------------------------------------------------------- #
def test_risk_agent_exposes_risk_level() -> None:
    result = registry.create("risk").run(_ctx())
    assert "risk_level" in result.extra
    assert result.extra["risk_level"] in {"low", "medium", "high"}
    # The agent also records the underlying volatility in range.
    assert "volatility" in result.extra
    assert 0.0 <= float(result.extra["volatility"]) <= 1.0


# --------------------------------------------------------------------------- #
# Expert agent: valid + explanation even without an LLM (offline fallback)
# --------------------------------------------------------------------------- #
def test_expert_agent_produces_explanation_without_llm() -> None:
    result = registry.create("expert").run(_ctx())

    assert result.agent == "expert"
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    # A grounded explanation is always present so predictions stay explainable.
    assert "explanation" in result.extra
    assert isinstance(result.extra["explanation"], str)
    assert result.extra["explanation"].strip()
    assert "recommendation" in result.extra
    assert result.extra["recommendation"] in {"positive", "negative"}
    # No LLM key in this environment -> deterministic fallback path.
    assert result.extra["llm_fallback"] is True


def test_expert_agent_consumes_peer_results() -> None:
    """The expert reads peer scores from ctx.context['peer_results']."""
    peers = [
        {"agent": "historical", "score": 0.8, "confidence": 0.7},
        {"agent": "trend", "score": 0.6, "confidence": 0.7},
    ]
    result = registry.create("expert").run(_ctx(context={"peer_results": peers}))
    # Score is the average of peer scores (0.8 + 0.6) / 2 = 0.7.
    assert result.score == pytest.approx(0.7, abs=1e-6)
    assert result.extra["recommendation"] == "positive"


def test_expert_agent_recommendation_follows_peers() -> None:
    low_peers = [
        {"agent": "historical", "score": 0.2, "confidence": 0.7},
        {"agent": "trend", "score": 0.1, "confidence": 0.7},
    ]
    result = registry.create("expert").run(_ctx(context={"peer_results": low_peers}))
    assert result.score == pytest.approx(0.15, abs=1e-6)
    assert result.extra["recommendation"] == "negative"


# --------------------------------------------------------------------------- #
# BaseAgent.run safety net
# --------------------------------------------------------------------------- #
class _BoomAgent(BaseAgent):
    name = "boom"

    def analyze(self, ctx: AgentContext) -> AgentResult:
        raise RuntimeError("intentional failure")


def test_failing_agent_returns_neutral_fallback_and_never_raises() -> None:
    agent = _BoomAgent()
    # run() must swallow the exception entirely.
    result = agent.run(_ctx())

    assert isinstance(result, AgentResult)
    assert result.agent == "boom"
    assert result.score == 0.5
    assert result.confidence == 0.0
    assert result.extra.get("error")


def test_base_agent_name_override_on_result() -> None:
    """run() rewrites a mismatched agent name to the agent's own name."""

    class _MislabelledAgent(BaseAgent):
        name = "correct"

        def analyze(self, ctx: AgentContext) -> AgentResult:
            return AgentResult(agent="wrong", score=0.4, confidence=0.5)

    result = _MislabelledAgent().run(_ctx())
    assert result.agent == "correct"


# --------------------------------------------------------------------------- #
# AgentResult validation (the normalised contract)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_score", [1.5, -0.1, 2.0])
def test_agent_result_rejects_out_of_range_score(bad_score: float) -> None:
    with pytest.raises(ValidationError):
        AgentResult(agent="x", score=bad_score, confidence=0.5)


@pytest.mark.parametrize("bad_conf", [1.5, -0.1])
def test_agent_result_rejects_out_of_range_confidence(bad_conf: float) -> None:
    with pytest.raises(ValidationError):
        AgentResult(agent="x", score=0.5, confidence=bad_conf)


def test_agent_result_accepts_boundary_values() -> None:
    lo = AgentResult(agent="x", score=0.0, confidence=0.0)
    hi = AgentResult(agent="x", score=1.0, confidence=1.0)
    assert lo.score == 0.0 and lo.confidence == 0.0
    assert hi.score == 1.0 and hi.confidence == 1.0
