"""Calibratable score-model parameters (Sprint WC-6).

The Poisson score model's two key constants (``base_goals`` and
``strength_sensitivity``) start from the hand-picked seed values in
:mod:`app.prediction.score` and can be **overridden by calibration**. The current
values are cached (Redis, with in-memory fallback) so the app and the scheduler
share one source — exactly like the agent weights.
"""

from __future__ import annotations

from app.cache import get_cache
from app.prediction.score import BASE_GOALS, STRENGTH_SENSITIVITY

_CACHE_KEY = "score:params"


def current_params() -> tuple[float, float]:
    """Return ``(base_goals, strength_sensitivity)`` — calibrated if available."""
    cached = get_cache().get(_CACHE_KEY)
    if isinstance(cached, dict):
        return (
            float(cached.get("base_goals", BASE_GOALS)),
            float(cached.get("strength_sensitivity", STRENGTH_SENSITIVITY)),
        )
    return BASE_GOALS, STRENGTH_SENSITIVITY


def set_params(base_goals: float, strength_sensitivity: float) -> None:
    get_cache().set(
        _CACHE_KEY,
        {"base_goals": base_goals, "strength_sensitivity": strength_sensitivity},
        ttl=30 * 24 * 3600,
    )
