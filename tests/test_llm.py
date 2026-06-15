"""Sprint 04 / Sprint 12 LLM-layer DoD tests.

Covers two contracts that must hold without any network or LLM credentials:

1. ``ModelRouter.select(complexity=...)`` honours its routing policy:
   - ``cost``     -> always the cheapest tier
   - ``quality``  -> always the highest-quality tier
   - ``balanced`` -> escalates to a higher (pricier) tier only when complexity
     is high, otherwise stays on the cheapest tier.

2. ``LLMProvider`` switches purely by config (``openai`` / ``ollama``) and
   degrades safely: with no client reachable, ``.available`` is False and
   ``.complete(prompt)`` returns a deterministic fallback ``LLMResponse`` that
   carries the configured provider and some non-empty text.

Expected tiers are derived from ``DEFAULT_TIERS`` (cheapest / highest-quality)
rather than hard-coded names, so the tests assert the policy contract itself.
"""

from __future__ import annotations

import pytest

from app.llm.provider import LLMProvider, LLMResponse
from app.llm.router import DEFAULT_TIERS, ModelRouter, ModelTier


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #
def _cheapest(tiers: list[ModelTier]) -> ModelTier:
    return min(tiers, key=lambda t: t.cost)


def _highest_quality(tiers: list[ModelTier]) -> ModelTier:
    return max(tiers, key=lambda t: t.quality)


def _priciest(tiers: list[ModelTier]) -> ModelTier:
    return max(tiers, key=lambda t: t.cost)


@pytest.fixture
def offline_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force Ollama unreachable so degradation tests stay deterministic even when
    a real ollama container happens to be running locally (the provider now pings
    the server at construction via ``validate_model_on_init``)."""
    monkeypatch.setattr("app.llm.provider.settings.ollama_base_url", "http://127.0.0.1:1")


# --------------------------------------------------------------------------- #
# Router: tier table sanity
# --------------------------------------------------------------------------- #
def test_default_tiers_are_non_trivial() -> None:
    # The escalation logic only has an effect with >1 tier, and the cheapest
    # tier must genuinely differ from the priciest for the contract to bite.
    assert len(DEFAULT_TIERS) >= 2
    assert _cheapest(DEFAULT_TIERS).cost < _priciest(DEFAULT_TIERS).cost


# --------------------------------------------------------------------------- #
# Router: "cost" policy -> always cheapest
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("complexity", [0.0, 0.3, 0.6, 0.9, 1.0])
def test_cost_policy_always_picks_cheapest(complexity: float) -> None:
    router = ModelRouter(policy="cost")
    chosen = router.select(complexity=complexity)
    cheapest = _cheapest(DEFAULT_TIERS)
    assert isinstance(chosen, ModelTier)
    assert chosen.name == cheapest.name
    assert chosen.cost == cheapest.cost
    # Cheapest means no other tier is cheaper.
    assert chosen.cost == min(t.cost for t in DEFAULT_TIERS)


def test_cost_policy_is_case_insensitive() -> None:
    # ModelRouter lowercases the policy on construction.
    router = ModelRouter(policy="COST")
    assert router.policy == "cost"
    assert router.select(complexity=0.95).name == _cheapest(DEFAULT_TIERS).name


# --------------------------------------------------------------------------- #
# Router: "quality" policy -> always highest quality
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("complexity", [0.0, 0.3, 0.6, 0.9, 1.0])
def test_quality_policy_always_picks_highest_quality(complexity: float) -> None:
    router = ModelRouter(policy="quality")
    chosen = router.select(complexity=complexity)
    best = _highest_quality(DEFAULT_TIERS)
    assert chosen.name == best.name
    assert chosen.quality == best.quality
    assert chosen.quality == max(t.quality for t in DEFAULT_TIERS)


# --------------------------------------------------------------------------- #
# Router: "balanced" policy -> escalate only when complex
# --------------------------------------------------------------------------- #
def test_balanced_policy_stays_cheap_on_low_complexity() -> None:
    router = ModelRouter(policy="balanced")
    chosen = router.select(complexity=0.1)
    assert chosen.name == _cheapest(DEFAULT_TIERS).name


def test_balanced_policy_escalates_on_high_complexity() -> None:
    router = ModelRouter(policy="balanced")
    chosen = router.select(complexity=0.9)
    # Escalation goes to the priciest tier (ordered[-1] by cost).
    assert chosen.name == _priciest(DEFAULT_TIERS).name
    # And that is strictly pricier than the low-complexity choice.
    low = router.select(complexity=0.1)
    assert chosen.cost > low.cost


def test_balanced_policy_threshold_is_point_six() -> None:
    # select() escalates on complexity >= 0.6; just below stays cheap.
    router = ModelRouter(policy="balanced")
    cheapest = _cheapest(DEFAULT_TIERS)
    priciest = _priciest(DEFAULT_TIERS)
    assert router.select(complexity=0.59).name == cheapest.name
    assert router.select(complexity=0.60).name == priciest.name


def test_balanced_default_complexity_stays_cheap() -> None:
    # Default complexity is 0.5 (< 0.6) so balanced should not escalate.
    router = ModelRouter(policy="balanced")
    assert router.select().name == _cheapest(DEFAULT_TIERS).name


def test_balanced_with_single_tier_does_not_escalate() -> None:
    # With only one tier there is nothing to escalate to, regardless of complexity.
    only = ModelTier(name="solo", model="m", cost=0.5, quality=0.8)
    router = ModelRouter(policy="balanced", tiers=[only])
    assert router.select(complexity=1.0).name == "solo"
    assert router.select(complexity=0.0).name == "solo"


def test_select_returns_a_member_of_configured_tiers() -> None:
    for policy in ("cost", "quality", "balanced"):
        router = ModelRouter(policy=policy)
        chosen = router.select(complexity=0.5)
        assert chosen in DEFAULT_TIERS


# --------------------------------------------------------------------------- #
# Provider: config-driven switching + safe degradation (OpenAI)
# --------------------------------------------------------------------------- #
def test_openai_provider_unavailable_without_key() -> None:
    # No API key is configured in the test env, so the client cannot be built.
    provider = LLMProvider(provider="openai", model="gpt-4o-mini")
    assert provider.provider == "openai"
    assert provider.available is False


def test_openai_complete_returns_deterministic_fallback() -> None:
    provider = LLMProvider(provider="openai", model="gpt-4o-mini")
    resp = provider.complete("Will it rain tomorrow?")
    assert isinstance(resp, LLMResponse)
    assert resp.fallback is True
    # Fallback carries the *configured* provider (proves config switching).
    assert resp.provider == "openai"
    # Non-empty, deterministic text — never asserting on live LLM output.
    assert isinstance(resp.text, str)
    assert resp.text.strip() != ""
    assert "fallback" in resp.text.lower()


def test_openai_fallback_is_deterministic_across_runs() -> None:
    provider = LLMProvider(provider="openai", model="gpt-4o-mini")
    first = provider.complete("same prompt")
    second = provider.complete("same prompt")
    assert (first.text, first.provider, first.model, first.fallback) == (
        second.text,
        second.provider,
        second.model,
        second.fallback,
    )


# --------------------------------------------------------------------------- #
# Provider: config-driven switching + safe degradation (Ollama)
# --------------------------------------------------------------------------- #
def test_ollama_provider_unavailable_without_server(offline_ollama: None) -> None:
    # No reachable Ollama server / client in the test env -> unavailable.
    provider = LLMProvider(provider="ollama", model="llama3")
    assert provider.provider == "ollama"
    assert provider.available is False


def test_ollama_complete_returns_deterministic_fallback(offline_ollama: None) -> None:
    provider = LLMProvider(provider="ollama", model="llama3")
    resp = provider.complete("Predict the outcome.")
    assert isinstance(resp, LLMResponse)
    assert resp.fallback is True
    # Fallback carries the *configured* provider, proving the provider switched
    # by config (openai vs ollama) rather than being hard-wired.
    assert resp.provider == "ollama"
    assert isinstance(resp.text, str)
    assert resp.text.strip() != ""


def test_provider_switch_is_reflected_in_fallback_provider_field(offline_ollama: None) -> None:
    # Same model, different configured provider -> distinct fallback provider.
    openai_resp = LLMProvider(provider="openai", model="shared-model").complete("x")
    ollama_resp = LLMProvider(provider="ollama", model="shared-model").complete("x")
    assert openai_resp.provider == "openai"
    assert ollama_resp.provider == "ollama"
    assert openai_resp.provider != ollama_resp.provider
    # Both degrade safely.
    assert openai_resp.fallback is True
    assert ollama_resp.fallback is True


def test_provider_normalises_provider_name_case() -> None:
    provider = LLMProvider(provider="OpenAI", model="gpt-4o-mini")
    assert provider.provider == "openai"


def test_unknown_provider_degrades_safely() -> None:
    # An unrecognised provider must not raise; it must fall back like the rest.
    provider = LLMProvider(provider="does-not-exist", model="m")
    assert provider.available is False
    resp = provider.complete("hello")
    assert resp.fallback is True
    assert resp.provider == "does-not-exist"
    assert resp.text.strip() != ""


# --------------------------------------------------------------------------- #
# Router <-> Provider integration: complete() never raises, always falls back
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("policy", ["cost", "quality", "balanced"])
def test_router_complete_returns_fallback_response(policy: str) -> None:
    router = ModelRouter(policy=policy)
    resp = router.complete("Estimate the result.", complexity=0.9)
    assert isinstance(resp, LLMResponse)
    # No credentials / server in the test env -> deterministic fallback.
    assert resp.fallback is True
    assert resp.text.strip() != ""
