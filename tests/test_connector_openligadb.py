"""Offline tests for the OpenLigaDB football connector (Sprint 08).

Exercise the pure parsing/metric/matchup functions with fixtures (no network)
and the agents' real-data path + heuristic fallback via a fake connector.
"""

from __future__ import annotations

from app.agents.base import AgentContext
from app.agents.registry import registry
from app.connectors.openligadb import (
    MatchupMetrics,
    matchup_from,
    metrics_from,
    momentum_from_form,
    opponent_team,
    recent_form,
    resolve_team,
    subject_team,
)

TABLE = [
    {
        "teamName": "FC Bayern München",
        "shortName": "Bayern",
        "points": 89,
        "won": 28,
        "draw": 5,
        "lost": 1,
        "matches": 34,
        "goalDiff": 86,
    },
    {
        "teamName": "Borussia Dortmund",
        "shortName": "BVB",
        "points": 73,
        "won": 22,
        "draw": 7,
        "lost": 5,
        "matches": 34,
        "goalDiff": 36,
    },
]

B = "FC Bayern München"
D = "Borussia Dortmund"


def _match(date: str, home: str, away: str, hs: int, as_: int) -> dict:
    return {
        "matchIsFinished": True,
        "matchDateTime": date,
        "team1": {"teamName": home},
        "team2": {"teamName": away},
        "matchResults": [{"resultTypeID": 2, "pointsTeam1": hs, "pointsTeam2": as_}],
    }


MATCHES = [
    _match("2026-04-01", B, "VfB Stuttgart", 0, 1),  # Bayern L
    _match("2026-04-08", "FC Augsburg", B, 1, 1),  # Bayern D
    _match("2026-04-15", B, "RB Leipzig", 3, 0),  # Bayern W
    _match("2026-04-22", "VfL Wolfsburg", B, 0, 1),  # Bayern W
    _match("2026-04-29", B, "1. FC Köln", 5, 1),  # Bayern W
    _match("2026-04-15", D, "Mainz 05", 2, 0),  # Dortmund W
    _match("2026-04-22", "Werder Bremen", D, 1, 0),  # Dortmund L
    _match("2026-04-29", D, "FC Augsburg", 1, 1),  # Dortmund D
]


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_subject_and_opponent_parsing() -> None:
    assert subject_team("Bayern vs Dortmund") == "Bayern"
    assert opponent_team("Bayern vs Dortmund") == "Dortmund"
    assert subject_team("Arsenal") == "Arsenal"
    assert opponent_team("Arsenal") == ""


def test_resolve_team_normalises_accents_and_substrings() -> None:
    assert resolve_team(TABLE, "Bayern")[0] == 0
    assert resolve_team(TABLE, "FC Bayern München vs Dortmund")[0] == 0
    assert resolve_team(TABLE, "Dortmund")[0] == 1
    assert resolve_team(TABLE, "Real Madrid") is None


def test_recent_form_and_momentum() -> None:
    form = recent_form(MATCHES, B)
    assert form == ["L", "D", "W", "W", "W"]  # oldest -> newest
    momentum = momentum_from_form(form)
    assert momentum is not None and momentum > 0.6
    assert momentum_from_form([]) is None


def test_metrics_from_computes_real_signals() -> None:
    m = metrics_from(TABLE, MATCHES, "Bayern")
    assert m is not None
    assert m.team == B and m.rank == 1 and m.played == 34
    assert m.win_rate == round(28 / 34, 3)
    assert m.strength == round(89 / (34 * 3), 3)
    assert metrics_from(TABLE, MATCHES, "Real Madrid") is None


# --------------------------------------------------------------------------- #
# Head-to-head
# --------------------------------------------------------------------------- #
def test_matchup_relative_favours_stronger_home() -> None:
    m = matchup_from(TABLE, MATCHES, "Bayern vs Dortmund")
    assert isinstance(m, MatchupMetrics)
    assert m.home.team == B and m.away is not None and m.away.team == D
    assert m.label == f"{B} vs {D}"
    # Bayern is clearly stronger -> all relative edges favour the home side.
    assert m.rel_win_rate > 0.5
    assert m.rel_strength > 0.5


def test_matchup_is_symmetric_when_reversed() -> None:
    ab = matchup_from(TABLE, MATCHES, "Bayern vs Dortmund")
    ba = matchup_from(TABLE, MATCHES, "Dortmund vs Bayern")
    assert ba.rel_win_rate < 0.5  # Dortmund (home) is the weaker side
    # Relative edges are symmetric: A-vs-B and B-vs-A sum to ~1.
    assert abs(ab.rel_win_rate + ba.rel_win_rate - 1.0) < 0.01
    assert abs(ab.rel_strength + ba.rel_strength - 1.0) < 0.01


def test_matchup_falls_back_to_absolute_for_single_team() -> None:
    m = matchup_from(TABLE, MATCHES, "Bayern")
    assert m.away is None
    assert m.rel_win_rate == m.home.win_rate
    assert m.rel_strength == m.home.strength


def test_matchup_unknown_home_returns_none() -> None:
    assert matchup_from(TABLE, MATCHES, "Real Madrid vs Barcelona") is None


# --------------------------------------------------------------------------- #
# Agent integration: real-data path vs heuristic fallback
# --------------------------------------------------------------------------- #
class _FakeConnector:
    domain = "sports"

    def __init__(self, matchup: MatchupMetrics | None) -> None:
        self._matchup = matchup

    def matchup_metrics(self, entity: str) -> MatchupMetrics | None:
        return self._matchup


def test_historical_agent_uses_real_matchup() -> None:
    matchup = matchup_from(TABLE, MATCHES, "Bayern vs Dortmund")
    ctx = AgentContext(
        entity="Bayern vs Dortmund", domain="sports", connectors=[_FakeConnector(matchup)]
    )
    result = registry.create("historical").run(ctx)
    assert result.score == matchup.rel_win_rate
    assert result.extra.get("source") == "openligadb"
    assert result.extra.get("home") == B
    assert result.extra.get("away") == D


def test_agents_fall_back_to_heuristic_without_connector() -> None:
    ctx = AgentContext(entity="Bayern vs Dortmund", domain="sports", connectors=[])
    for name in ("historical", "trend", "market"):
        result = registry.create(name).run(ctx)
        assert 0.0 <= result.score <= 1.0
        assert result.extra.get("source") is None


def test_risk_uses_real_uncertainty_and_contextual_abstains() -> None:
    """WC-7: Risk derives a real uncertainty signal from any resolvable matchup;
    Contextual (host advantage only) abstains for non-host sports matches."""
    matchup = matchup_from(TABLE, MATCHES, "Bayern vs Dortmund")
    ctx = AgentContext(
        entity="Bayern vs Dortmund", domain="sports", connectors=[_FakeConnector(matchup)]
    )

    risk = registry.create("risk").run(ctx)
    assert risk.extra.get("source") == "openligadb"
    assert risk.extra.get("risk_level") in {"low", "medium", "high"}
    assert risk.extra.get("abstained") is None

    contextual = registry.create("contextual").run(ctx)
    assert contextual.score == 0.5
    assert contextual.extra.get("abstained") is True


def test_unwired_agents_use_heuristic_without_connector() -> None:
    """Without a connector (e.g. non-sports / tests) they keep their heuristic."""
    ctx = AgentContext(entity="Acme", domain="sports", connectors=[])
    for name in ("contextual", "risk"):
        result = registry.create(name).run(ctx)
        assert result.extra.get("abstained") is None
