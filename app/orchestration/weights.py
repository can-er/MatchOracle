"""Agent weighting (Sprint 05) with dynamic recalculation hooks (Sprint 10).

Weights start from the configured defaults. Sprint 10 adjusts them from measured
accuracy with guardrails (clamping + smoothing) to avoid brutal regressions.
The current weight vector is cached in Redis so workers share one source.
"""

from __future__ import annotations

from app.cache import get_cache
from app.config import DEFAULT_WEIGHTS, settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_CACHE_KEY = "orchestration:weights"


class WeightManager:
    """Owns the agent weight vector and its normalisation/adjustment."""

    def __init__(self) -> None:
        self._cache = get_cache()

    def current(self) -> dict[str, float]:
        cached = self._cache.get(_CACHE_KEY)
        if isinstance(cached, dict) and cached:
            return {k: float(v) for k, v in cached.items()}
        return dict(settings.agent_weights or DEFAULT_WEIGHTS)

    def set(self, weights: dict[str, float]) -> dict[str, float]:
        normalised = self.normalise(weights)
        self._cache.set(_CACHE_KEY, normalised, ttl=86_400)
        return normalised

    def reset(self) -> dict[str, float]:
        self._cache.delete(_CACHE_KEY)
        return self.current()

    @staticmethod
    def normalise(weights: dict[str, float]) -> dict[str, float]:
        total = sum(max(0.0, w) for w in weights.values())
        if total <= 0:
            n = len(weights) or 1
            return dict.fromkeys(weights, 1.0 / n)
        return {k: round(max(0.0, w) / total, 4) for k, w in weights.items()}

    def adjust_from_accuracy(self, accuracy: dict[str, float]) -> dict[str, float]:
        """Nudge weights toward agents with higher accuracy (Sprint 10).

        Uses a bounded learning rate plus min/max clamps so a single bad window
        cannot collapse or dominate the vector.
        """
        if not settings.autoweight_enabled or not accuracy:
            return self.current()

        lr = settings.autoweight_learning_rate
        current = self.current()
        mean_acc = sum(accuracy.values()) / len(accuracy)
        updated: dict[str, float] = {}
        for agent, w in current.items():
            acc = accuracy.get(agent, mean_acc)
            # Multiplicative nudge proportional to (acc - mean).
            factor = 1.0 + lr * (acc - mean_acc)
            new_w = w * factor
            updated[agent] = min(settings.autoweight_max, max(settings.autoweight_min, new_w))

        normalised = self.set(updated)
        logger.info("orchestration.weights.adjusted", weights=normalised)
        return normalised


def get_weight_manager() -> WeightManager:
    return WeightManager()
