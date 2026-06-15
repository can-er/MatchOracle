from __future__ import annotations

import math

import pytest

from app.config import DEFAULT_WEIGHTS, settings
from app.orchestration.weights import _CACHE_KEY, WeightManager, get_weight_manager


@pytest.fixture
def manager() -> WeightManager:
    """A fresh WeightManager with an isolated cache key.

    ``get_cache()`` is an lru_cache'd process-global, so the underlying cache is
    shared across instances. Clear the orchestration key before and after each
    test so state never leaks between tests.
    """
    wm = WeightManager()
    wm._cache.delete(_CACHE_KEY)
    try:
        yield wm
    finally:
        wm._cache.delete(_CACHE_KEY)


# --- normalise -----------------------------------------------------------------


def test_normalise_sums_to_one(manager: WeightManager) -> None:
    out = manager.normalise({"a": 2.0, "b": 3.0, "c": 5.0})
    assert math.isclose(sum(out.values()), 1.0, abs_tol=1e-3)
    assert set(out) == {"a", "b", "c"}
    assert all(isinstance(v, float) for v in out.values())


def test_normalise_preserves_relative_proportions(manager: WeightManager) -> None:
    out = manager.normalise({"a": 1.0, "b": 3.0})
    # b should be roughly 3x a after normalisation.
    assert out["b"] > out["a"]
    assert math.isclose(out["b"] / out["a"], 3.0, rel_tol=1e-2)


def test_normalise_clamps_negatives_to_zero(manager: WeightManager) -> None:
    out = manager.normalise({"a": -5.0, "b": 1.0, "c": 1.0})
    assert out["a"] == 0.0
    assert all(v >= 0.0 for v in out.values())
    # The two positive weights should split the mass evenly.
    assert math.isclose(out["b"], out["c"], abs_tol=1e-3)
    assert math.isclose(sum(out.values()), 1.0, abs_tol=1e-3)


def test_normalise_all_zero_yields_uniform(manager: WeightManager) -> None:
    out = manager.normalise({"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0})
    assert math.isclose(sum(out.values()), 1.0, abs_tol=1e-9)
    assert all(math.isclose(v, 0.25, abs_tol=1e-9) for v in out.values())


def test_normalise_all_negative_yields_uniform(manager: WeightManager) -> None:
    # Total clamped mass is <= 0 -> uniform fallback.
    out = manager.normalise({"x": -1.0, "y": -2.0})
    assert all(math.isclose(v, 0.5, abs_tol=1e-9) for v in out.values())
    assert math.isclose(sum(out.values()), 1.0, abs_tol=1e-9)


def test_normalise_empty_does_not_crash(manager: WeightManager) -> None:
    # len(weights) or 1 guards division-by-zero; empty stays empty.
    assert manager.normalise({}) == {}


# --- current / set / reset -----------------------------------------------------


def test_current_returns_defaults_when_uncached(manager: WeightManager) -> None:
    out = manager.current()
    assert out == dict(settings.agent_weights or DEFAULT_WEIGHTS)
    # The configured defaults match DEFAULT_WEIGHTS in this environment.
    assert out == dict(DEFAULT_WEIGHTS)
    assert set(out) == set(DEFAULT_WEIGHTS)


def test_set_then_current_returns_normalised(manager: WeightManager) -> None:
    raw = {
        "historical": 4.0,
        "trend": 4.0,
        "contextual": 0.0,
        "risk": 0.0,
        "market": 0.0,
        "expert": 0.0,
    }
    returned = manager.set(raw)
    # set() returns the normalised vector.
    assert math.isclose(sum(returned.values()), 1.0, abs_tol=1e-3)
    # current() now reflects the cached normalised set.
    cur = manager.current()
    assert cur == returned
    assert math.isclose(cur["historical"], 0.5, abs_tol=1e-3)
    assert math.isclose(cur["trend"], 0.5, abs_tol=1e-3)
    assert cur["contextual"] == 0.0


def test_set_persists_across_new_manager_instances(manager: WeightManager) -> None:
    # The cache is a shared process-global, so a new manager sees the same state.
    manager.set({"a": 1.0, "b": 1.0})
    other = WeightManager()
    assert math.isclose(other.current()["a"], 0.5, abs_tol=1e-3)


def test_reset_restores_defaults(manager: WeightManager) -> None:
    manager.set(
        {
            "historical": 1.0,
            "trend": 0.0,
            "contextual": 0.0,
            "risk": 0.0,
            "market": 0.0,
            "expert": 0.0,
        }
    )
    assert manager.current() != dict(DEFAULT_WEIGHTS)
    restored = manager.reset()
    assert restored == dict(settings.agent_weights or DEFAULT_WEIGHTS)
    assert restored == dict(DEFAULT_WEIGHTS)
    # And current() agrees afterwards.
    assert manager.current() == dict(DEFAULT_WEIGHTS)


def test_get_weight_manager_returns_instance() -> None:
    assert isinstance(get_weight_manager(), WeightManager)


# --- adjust_from_accuracy ------------------------------------------------------


def test_adjust_from_accuracy_sums_to_one(manager: WeightManager) -> None:
    accuracy = {
        "historical": 0.9,
        "trend": 0.5,
        "contextual": 0.5,
        "risk": 0.5,
        "market": 0.5,
        "expert": 0.1,
    }
    out = manager.adjust_from_accuracy(accuracy)
    assert math.isclose(sum(out.values()), 1.0, abs_tol=1e-3)
    assert set(out) == set(DEFAULT_WEIGHTS)
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_adjust_favours_more_accurate_agent(manager: WeightManager) -> None:
    # historical and expert start with weights 0.25 and 0.15 respectively.
    # Give the lower-weighted 'expert' the higher accuracy and confirm the nudge
    # still respects the ordering of accuracy on equally-weighted agents instead.
    # Use two agents with identical starting weights for a clean comparison.
    accuracy = {
        "contextual": 0.95,  # clearly more accurate
        "risk": 0.05,  # clearly less accurate
        "historical": 0.5,
        "trend": 0.5,
        "market": 0.5,
        "expert": 0.5,
    }
    base = manager.current()
    assert math.isclose(base["contextual"], base["risk"], abs_tol=1e-9)
    out = manager.adjust_from_accuracy(accuracy)
    # The clearly-more-accurate agent ends with weight >= the less-accurate one.
    assert out["contextual"] >= out["risk"]
    assert out["contextual"] > out["risk"]


def test_adjust_respects_per_agent_clamp_before_renormalise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-agent values are bounded by [autoweight_min, autoweight_max].

    We assert the pre-renormalisation clamp by reconstructing the intermediate
    vector with the same formula the manager uses, then confirm the manager's
    final (renormalised) output is consistent with a clamped intermediate.
    """
    wm = WeightManager()
    wm._cache.delete(_CACHE_KEY)
    try:
        # Extreme accuracies that would otherwise push some agents far past the
        # max / below the min if unclamped.
        accuracy = {name: (5.0 if name == "historical" else -5.0) for name in DEFAULT_WEIGHTS}
        current = wm.current()
        lr = settings.autoweight_learning_rate
        mean_acc = sum(accuracy.values()) / len(accuracy)
        intermediate: dict[str, float] = {}
        for agent, w in current.items():
            acc = accuracy[agent]
            new_w = w * (1.0 + lr * (acc - mean_acc))
            clamped = min(settings.autoweight_max, max(settings.autoweight_min, new_w))
            intermediate[agent] = clamped
            # Every intermediate (pre-normalise) value sits within the bounds.
            assert settings.autoweight_min <= clamped <= settings.autoweight_max

        out = wm.adjust_from_accuracy(accuracy)
        # Final result equals normalise() of the clamped intermediate.
        assert out == wm.normalise(intermediate)
        assert math.isclose(sum(out.values()), 1.0, abs_tol=1e-3)
    finally:
        wm._cache.delete(_CACHE_KEY)


def test_adjust_disabled_returns_current(
    monkeypatch: pytest.MonkeyPatch, manager: WeightManager
) -> None:
    monkeypatch.setattr(settings, "autoweight_enabled", False)
    out = manager.adjust_from_accuracy({"historical": 0.99})
    assert out == manager.current()
    # Nothing was cached as a side effect (still defaults).
    assert out == dict(DEFAULT_WEIGHTS)


def test_adjust_empty_accuracy_returns_current(manager: WeightManager) -> None:
    out = manager.adjust_from_accuracy({})
    assert out == manager.current()


def test_adjust_is_deterministic(manager: WeightManager) -> None:
    accuracy = {
        "historical": 0.8,
        "trend": 0.6,
        "contextual": 0.4,
        "risk": 0.4,
        "market": 0.4,
        "expert": 0.2,
    }
    first = manager.adjust_from_accuracy(accuracy)
    manager.reset()
    second = manager.adjust_from_accuracy(accuracy)
    assert first == second
