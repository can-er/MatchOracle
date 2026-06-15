"""Score-model calibration (Sprint WC-6).

Once enough matches have been played, fit the Poisson model's two key constants
(``base_goals``, ``strength_sensitivity``) to the real results by **maximum
likelihood** — the values that make the actual scorelines most probable. A simple
grid search keeps it dependency-free and the grid bounds act as guardrails.

Samples are ``{home_strength, away_strength, home_goals, away_goals}`` dicts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MIN_SAMPLES = 12  # don't calibrate on noise

# Grid (also the guardrails — stays in a sane football range).
_BASE_GRID = [round(0.9 + 0.05 * i, 3) for i in range(19)]  # 0.90 .. 1.80
_SENS_GRID = [round(0.5 + 0.1 * i, 3) for i in range(26)]  # 0.5 .. 3.0


@dataclass
class Calibration:
    base_goals: float
    strength_sensitivity: float
    log_likelihood: float
    samples: int


def _log_poisson(k: int, lam: float) -> float:
    # log(e^-lam * lam^k / k!) = -lam + k*log(lam) - log(k!)
    return -lam + k * math.log(lam) - math.lgamma(k + 1)


def _sample_loglik(sample: dict, base_goals: float, sensitivity: float) -> float:
    gap = sample["home_strength"] - sample["away_strength"]  # neutral venue (World Cup)
    lam_home = base_goals * math.exp(sensitivity * gap * 0.5)
    lam_away = base_goals * math.exp(-sensitivity * gap * 0.5)
    return _log_poisson(sample["home_goals"], lam_home) + _log_poisson(
        sample["away_goals"], lam_away
    )


def calibrate(samples: list[dict], *, min_samples: int = MIN_SAMPLES) -> Calibration | None:
    """Fit ``(base_goals, strength_sensitivity)`` by MLE; None below min_samples."""
    if len(samples) < min_samples:
        return None
    best_base, best_sens, best_ll = _BASE_GRID[0], _SENS_GRID[0], float("-inf")
    for base in _BASE_GRID:
        for sens in _SENS_GRID:
            ll = sum(_sample_loglik(s, base, sens) for s in samples)
            if ll > best_ll:
                best_base, best_sens, best_ll = base, sens, ll
    return Calibration(
        base_goals=best_base,
        strength_sensitivity=best_sens,
        log_likelihood=round(best_ll, 3),
        samples=len(samples),
    )
