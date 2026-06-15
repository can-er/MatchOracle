"""Full-tournament Monte-Carlo: bracket & champion (Sprint WC-4).

Each simulation plays the whole World Cup: the 12 groups (sampling scorelines
from the Poisson model, real results locked in), takes the 12 winners + 12
runners-up + the 8 best third-placed teams, **seeds them into a 32-team
single-elimination bracket**, and plays it out (knockout = forced winner). Over
many runs this yields each team's probability of reaching every round and of
lifting the trophy.

The bracket is a standard *seeded* bracket (an approximation — the exact FIFA
slotting of the best thirds is a complex lookup; this seeds by group-stage
performance instead). Deterministic given the seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from app.prediction.group import _sample_poisson
from app.prediction.params import current_params
from app.prediction.score import expected_goals, shootout_prob

# Group standings entry: (team, points, goal_diff, goals_for).
_Standing = tuple[str, int, int, int]


@dataclass
class TeamOutlook:
    team: str
    p_r16: float  # reaches the round of 16 (wins its round-of-32 tie)
    p_qf: float
    p_sf: float
    p_final: float
    p_champion: float


@dataclass
class TournamentPrediction:
    champion: str
    p_champion: float
    outlook: list[TeamOutlook] = field(default_factory=list)  # sorted by p_champion


def _bracket_seed_order(n: int) -> list[int]:
    """Standard seeding order for a power-of-two bracket (seed 1 vs 2 in the final)."""
    seeds = [1]
    while len(seeds) < n:
        size = len(seeds) * 2 + 1
        seeds = [s for pair in ((x, size - x) for x in seeds) for s in pair]
    return seeds


def _play_group(
    names: list[str],
    fixtures: list[tuple[str, str]],
    results: dict[tuple[str, str], tuple[int, int]],
    lambdas: dict[tuple[str, str], tuple[float, float]],
    rng: random.Random,
) -> list[_Standing]:
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
        names, key=lambda t: (points[t], goal_diff[t], goals_for[t], rng.random()), reverse=True
    )
    return [(t, points[t], goal_diff[t], goals_for[t]) for t in ranked]


def _knockout_winner(
    home: str,
    away: str,
    strengths: dict[str, float],
    base_goals: float,
    sensitivity: float,
    rng: random.Random,
) -> str:
    lam_home, lam_away = expected_goals(
        strengths[home],
        strengths[away],
        neutral=True,
        base_goals=base_goals,
        strength_sensitivity=sensitivity,
    )
    hg, ag = _sample_poisson(lam_home, rng), _sample_poisson(lam_away, rng)
    if hg > ag:
        return home
    if ag > hg:
        return away
    # Level after regulation -> decided by the shootout (tilted to the stronger side).
    return home if rng.random() < shootout_prob(strengths[home], strengths[away]) else away


def simulate_tournament(
    all_groups: dict[str, tuple[dict[str, float], list[tuple[str, str]], dict]],
    *,
    sims: int = 2000,
    seed: int = 0,
) -> TournamentPrediction:
    """Simulate the whole tournament ``sims`` times from the 12 groups' data."""
    rng = random.Random(seed)
    base_goals, sensitivity = current_params()

    strengths: dict[str, float] = {}
    group_lambdas: dict[str, dict[tuple[str, str], tuple[float, float]]] = {}
    for group, (teams, fixtures, _results) in all_groups.items():
        strengths.update(teams)
        group_lambdas[group] = {
            (home, away): expected_goals(
                teams[home],
                teams[away],
                neutral=True,
                base_goals=base_goals,
                strength_sensitivity=sensitivity,
            )
            for home, away in fixtures
        }

    reached = {t: {"r16": 0, "qf": 0, "sf": 0, "final": 0, "champion": 0} for t in strengths}
    order = _bracket_seed_order(32)
    round_names = ["r16", "qf", "sf", "final", "champion"]

    for _ in range(sims):
        winners: list[_Standing] = []
        runners: list[_Standing] = []
        thirds: list[_Standing] = []
        for group, (teams, fixtures, results) in all_groups.items():
            standings = _play_group(list(teams), fixtures, results, group_lambdas[group], rng)
            winners.append(standings[0])
            runners.append(standings[1])
            thirds.append(standings[2])

        rank = lambda s: (s[1], s[2], s[3], rng.random())  # noqa: E731 — points, gd, gf
        best_thirds = sorted(thirds, key=rank, reverse=True)[:8]
        seeded = (
            [t[0] for t in sorted(winners, key=rank, reverse=True)]
            + [t[0] for t in sorted(runners, key=rank, reverse=True)]
            + [t[0] for t in best_thirds]
        )
        bracket = [seeded[s - 1] for s in order]

        round_index = 0
        while len(bracket) > 1:
            next_round = []
            for i in range(0, len(bracket), 2):
                winner = _knockout_winner(
                    bracket[i], bracket[i + 1], strengths, base_goals, sensitivity, rng
                )
                next_round.append(winner)
                reached[winner][round_names[round_index]] += 1
            bracket = next_round
            round_index += 1

    outlook = [
        TeamOutlook(
            team=t,
            p_r16=round(r["r16"] / sims, 3),
            p_qf=round(r["qf"] / sims, 3),
            p_sf=round(r["sf"] / sims, 3),
            p_final=round(r["final"] / sims, 3),
            p_champion=round(r["champion"] / sims, 3),
        )
        for t, r in reached.items()
    ]
    outlook.sort(key=lambda o: o.p_champion, reverse=True)
    champion = outlook[0].team if outlook else ""
    p_champion = outlook[0].p_champion if outlook else 0.0
    return TournamentPrediction(champion=champion, p_champion=p_champion, outlook=outlook)
