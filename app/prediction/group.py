"""Group-stage prediction via Monte-Carlo simulation (Sprint WC-3).

A World Cup group is a 4-team round-robin (6 matches). We simulate it many times
— sampling each unplayed match's scoreline from the Poisson goals model, and
using real results for matches already played — then tally how often each team
finishes 1st/2nd/3rd and qualifies (top 2). Deterministic given the seed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from app.prediction.params import current_params
from app.prediction.score import expected_goals


@dataclass
class TeamStanding:
    team: str
    exp_points: float
    p_first: float
    p_second: float
    p_third: float
    p_qualify: float  # finishes in the top 2


@dataclass
class GroupPrediction:
    group: str
    standings: list[TeamStanding] = field(default_factory=list)  # best -> worst
    qualifiers: list[str] = field(default_factory=list)  # predicted top 2


def _sample_poisson(lam: float, rng: random.Random) -> int:
    """Sample a Poisson(lam) count (Knuth's algorithm)."""
    target = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= target:
            return k - 1


def simulate_group(
    group: str,
    teams: dict[str, float],
    fixtures: list[tuple[str, str]],
    *,
    results: dict[tuple[str, str], tuple[int, int]] | None = None,
    sims: int = 10_000,
    seed: int = 0,
) -> GroupPrediction:
    """Simulate a group ``sims`` times and return per-team finish probabilities.

    ``teams`` maps team name -> strength in (0,1]; ``fixtures`` are the six
    ``(home, away)`` pairings; ``results`` carries actual scores for matches
    already played (those are used as-is instead of being sampled).
    """
    rng = random.Random(seed)
    names = list(teams)
    results = results or {}
    base_goals, strength_sensitivity = current_params()
    lambdas = {
        (home, away): expected_goals(
            teams[home],
            teams[away],
            neutral=True,
            base_goals=base_goals,
            strength_sensitivity=strength_sensitivity,
        )
        for home, away in fixtures
    }

    tally = {t: {"first": 0, "second": 0, "third": 0, "qualify": 0} for t in names}
    points_sum = dict.fromkeys(names, 0)

    for _ in range(sims):
        points = dict.fromkeys(names, 0)
        goal_diff = dict.fromkeys(names, 0)
        goals_for = dict.fromkeys(names, 0)
        for home, away in fixtures:
            if (home, away) in results:
                hg, ag = results[(home, away)]
            else:
                lam_home, lam_away = lambdas[(home, away)]
                hg, ag = _sample_poisson(lam_home, rng), _sample_poisson(lam_away, rng)
            goals_for[home] += hg
            goals_for[away] += ag
            goal_diff[home] += hg - ag
            goal_diff[away] += ag - hg
            if hg > ag:
                points[home] += 3
            elif hg < ag:
                points[away] += 3
            else:
                points[home] += 1
                points[away] += 1

        ranked = sorted(
            names,
            key=lambda t: (points[t], goal_diff[t], goals_for[t], rng.random()),
            reverse=True,
        )
        tally[ranked[0]]["first"] += 1
        tally[ranked[1]]["second"] += 1
        tally[ranked[2]]["third"] += 1
        tally[ranked[0]]["qualify"] += 1
        tally[ranked[1]]["qualify"] += 1
        for t in names:
            points_sum[t] += points[t]

    standings = [
        TeamStanding(
            team=t,
            exp_points=round(points_sum[t] / sims, 2),
            p_first=round(tally[t]["first"] / sims, 3),
            p_second=round(tally[t]["second"] / sims, 3),
            p_third=round(tally[t]["third"] / sims, 3),
            p_qualify=round(tally[t]["qualify"] / sims, 3),
        )
        for t in names
    ]
    standings.sort(key=lambda s: (s.p_qualify, s.exp_points), reverse=True)
    qualifiers = [standings[0].team, standings[1].team] if len(standings) >= 2 else []
    return GroupPrediction(group=group, standings=standings, qualifiers=qualifiers)
