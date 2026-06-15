"""Tests for the Monte-Carlo group-stage simulator (Sprint WC-3)."""

from __future__ import annotations

from app.prediction.group import simulate_group

TEAMS = {"A": 0.9, "B": 0.6, "C": 0.5, "D": 0.2}
# Standard 4-team round robin (6 matches).
FIXTURES = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "D"), ("B", "C")]


def test_simulate_group_ranks_by_strength() -> None:
    gp = simulate_group("GROUP_X", TEAMS, FIXTURES, sims=3000, seed=1)
    order = [s.team for s in gp.standings]
    assert order[0] == "A"  # strongest most likely to top the group
    assert order[-1] == "D"  # weakest most likely bottom
    assert gp.qualifiers == order[:2]
    assert gp.standings[0].p_qualify > gp.standings[-1].p_qualify
    for s in gp.standings:
        assert 0.0 <= s.p_qualify <= 1.0
    # Exactly one team finishes first, exactly two qualify, each simulation.
    assert abs(sum(s.p_first for s in gp.standings) - 1.0) < 0.02
    assert abs(sum(s.p_qualify for s in gp.standings) - 2.0) < 0.05


def test_simulate_group_is_deterministic() -> None:
    a = simulate_group("G", TEAMS, FIXTURES, sims=1000, seed=7)
    b = simulate_group("G", TEAMS, FIXTURES, sims=1000, seed=7)
    assert [(s.team, s.p_qualify) for s in a.standings] == [
        (s.team, s.p_qualify) for s in b.standings
    ]


def test_played_results_are_locked_in() -> None:
    # D actually wins all three of its matches 3-0 -> 9 points -> qualifies despite
    # being the weakest on paper.
    results = {("C", "D"): (0, 3), ("B", "D"): (0, 3), ("A", "D"): (0, 3)}
    gp = simulate_group("G", TEAMS, FIXTURES, results=results, sims=2000, seed=2)
    standing_d = next(s for s in gp.standings if s.team == "D")
    assert standing_d.p_qualify > 0.8
    assert "D" in gp.qualifiers
