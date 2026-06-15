"""Tests for the full-tournament Monte-Carlo (Sprint WC-4)."""

from __future__ import annotations

from app.prediction.tournament import _bracket_seed_order, simulate_tournament


def _make_groups() -> dict:
    """12 groups of 4; strengths decrease across and within groups (A1 strongest)."""
    groups = {}
    for gi, letter in enumerate("ABCDEFGHIJKL"):
        names = [f"{letter}{i}" for i in range(1, 5)]
        teams = {n: round(max(0.12, 0.95 - 0.015 * (gi * 4 + j)), 3) for j, n in enumerate(names)}
        fixtures = [
            (names[0], names[1]),
            (names[2], names[3]),
            (names[0], names[2]),
            (names[1], names[3]),
            (names[0], names[3]),
            (names[1], names[2]),
        ]
        groups[f"GROUP_{letter}"] = (teams, fixtures, {})
    return groups


def test_bracket_seed_order_is_a_valid_permutation() -> None:
    order = _bracket_seed_order(32)
    assert sorted(order) == list(range(1, 33))
    assert order[0] == 1  # top seed leads the bracket


def test_simulate_tournament_funnel_and_strong_favourite() -> None:
    pred = simulate_tournament(_make_groups(), sims=400, seed=1)
    assert pred.p_champion > 0
    assert pred.champion == pred.outlook[0].team

    # Per team, the reach probabilities only shrink deeper into the bracket.
    for o in pred.outlook:
        assert o.p_r16 >= o.p_qf >= o.p_sf >= o.p_final >= o.p_champion

    # Exactly 16 teams reach the round of 16, exactly one champion, each simulation.
    assert abs(sum(o.p_r16 for o in pred.outlook) - 16.0) < 0.6
    assert abs(sum(o.p_champion for o in pred.outlook) - 1.0) < 0.05

    # The globally strongest team is a genuine title favourite.
    a1 = next(o for o in pred.outlook if o.team == "A1")
    assert a1.p_champion > 0.05
