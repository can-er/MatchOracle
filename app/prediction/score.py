"""Final-score prediction via a Poisson goals model (World Cup).

The standard approach in football analytics: model each team's goals as
Poisson-distributed around an **expected-goals** value (λ) derived from the two
teams' relative strength. From the joint score grid we read off:

- the **most likely exact scoreline** (the mode of the grid),
- the **top scorelines** with probabilities,
- the **outcome probabilities** (home win / draw / away win).

Strengths are the [0,1] values the connectors already produce (FIFA ranking,
later blended with live form), so this plugs straight into the matchup pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# World Cup matches average ~2.6 goals. These are hand-picked "seed" values.
# TODO(WC-6): calibrate against real results (vault: "Sprint WC-6 - Calibration").
BASE_GOALS = 1.30  # expected goals for an evenly-matched side
STRENGTH_SENSITIVITY = 1.8  # how strongly the strength gap shifts expected goals
HOME_ADVANTAGE = 0.08  # small bump for the first-named side (mostly neutral venues)
MAX_GOALS = 7  # truncate the Poisson grid


@dataclass
class ScorePrediction:
    home_goals: int
    away_goals: int
    scoreline: str
    exp_home: float
    exp_away: float
    p_home_win: float
    p_draw: float
    p_away_win: float
    top_scorelines: list[dict] = field(default_factory=list)
    # Knockout only: a regulation draw is decided by extra time / penalties, so a
    # winner is forced. ``winner`` is "home"/"away" (the side that advances).
    knockout: bool = False
    winner: str | None = None
    p_home_advance: float | None = None
    p_away_advance: float | None = None
    decided_by: str | None = None  # "regulation" or "shootout"


def shootout_prob(home_strength: float, away_strength: float) -> float:
    """P(home wins a tie-breaker). Shootouts are near coin-flips, mildly tilted
    toward the stronger side."""
    return round(min(0.75, max(0.25, 0.5 + 0.25 * (home_strength - away_strength))), 3)


def expected_goals(
    home_strength: float,
    away_strength: float,
    *,
    neutral: bool = False,
    base_goals: float = BASE_GOALS,
    strength_sensitivity: float = STRENGTH_SENSITIVITY,
) -> tuple[float, float]:
    """Expected goals (λ) for each side from their strengths (both in (0,1]).

    ``base_goals`` / ``strength_sensitivity`` default to the seed constants but
    can be overridden (used by calibration — Sprint WC-6).
    """
    gap = home_strength - away_strength
    home_adv = 0.0 if neutral else HOME_ADVANTAGE
    lam_home = base_goals * math.exp(strength_sensitivity * gap * 0.5 + home_adv)
    lam_away = base_goals * math.exp(-strength_sensitivity * gap * 0.5)
    return round(lam_home, 3), round(lam_away, 3)


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def predict_scoreline(
    home_strength: float,
    away_strength: float,
    *,
    neutral: bool = False,
    knockout: bool = False,
    max_goals: int = MAX_GOALS,
) -> ScorePrediction:
    """Predict the most likely final score and outcome probabilities.

    When ``knockout`` is True a draw can't stand: a winner is forced via an extra
    time / penalty tie-breaker, and ``winner`` / ``p_*_advance`` are filled in.
    """
    from app.prediction.params import current_params

    base_goals, strength_sensitivity = current_params()
    lam_home, lam_away = expected_goals(
        home_strength,
        away_strength,
        neutral=neutral,
        base_goals=base_goals,
        strength_sensitivity=strength_sensitivity,
    )

    grid: dict[tuple[int, int], float] = {}
    p_home = p_draw = p_away = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob = _poisson_pmf(i, lam_home) * _poisson_pmf(j, lam_away)
            grid[(i, j)] = prob
            if i > j:
                p_home += prob
            elif i == j:
                p_draw += prob
            else:
                p_away += prob

    total = sum(grid.values()) or 1.0  # truncation loses a sliver of mass
    best = max(grid, key=lambda k: grid[k])
    top = sorted(grid.items(), key=lambda kv: kv[1], reverse=True)[:5]
    p_home_n = round(p_home / total, 3)
    p_draw_n = round(p_draw / total, 3)
    p_away_n = round(p_away / total, 3)

    winner: str | None = None
    p_home_adv: float | None = None
    p_away_adv: float | None = None
    decided_by: str | None = None
    if knockout:
        p_shootout = shootout_prob(home_strength, away_strength)
        adv_home = p_home / total + (p_draw / total) * p_shootout
        adv_away = p_away / total + (p_draw / total) * (1.0 - p_shootout)
        p_home_adv = round(adv_home, 3)
        p_away_adv = round(adv_away, 3)
        winner = "home" if adv_home >= adv_away else "away"
        decided_by = "shootout" if best[0] == best[1] else "regulation"

    return ScorePrediction(
        home_goals=best[0],
        away_goals=best[1],
        scoreline=f"{best[0]}-{best[1]}",
        exp_home=lam_home,
        exp_away=lam_away,
        p_home_win=p_home_n,
        p_draw=p_draw_n,
        p_away_win=p_away_n,
        top_scorelines=[{"score": f"{i}-{j}", "p": round(prob / total, 3)} for (i, j), prob in top],
        knockout=knockout,
        winner=winner,
        p_home_advance=p_home_adv,
        p_away_advance=p_away_adv,
        decided_by=decided_by,
    )
