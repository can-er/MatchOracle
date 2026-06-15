"""Offline tests for the World Cup connector (Sprint WC-1).

The FIFA-ranking seed needs no network/key, so everything here runs offline.
"""

from __future__ import annotations

from app.agents.base import AgentContext
from app.agents.registry import registry
from app.connectors.worldcup import (
    WorldCupConnector,
    enrich_matchup_with_form,
    matchup_from_ranking,
    resolve_team,
    strength_of,
    wc_form_from_matches,
)


def _wc_match(date: str, home: str, away: str, hs: int, as_: int, status: str = "FINISHED") -> dict:
    return {
        "status": status,
        "utcDate": date,
        "homeTeam": {"name": home},
        "awayTeam": {"name": away},
        "score": {"fullTime": {"home": hs, "away": as_}},
    }


WC_MATCHES = [
    _wc_match("2026-06-12T18:00:00Z", "Argentina", "Nigeria", 2, 0),  # Argentina W
    _wc_match("2026-06-18T18:00:00Z", "Canada", "Argentina", 1, 1),  # Argentina D
    _wc_match("2026-06-24T18:00:00Z", "Argentina", "Croatia", 0, 1),  # Argentina L
    _wc_match("2026-06-12T18:00:00Z", "Saudi Arabia", "Mexico", 0, 0, status="TIMED"),  # ignored
]


def test_resolve_team_and_aliases() -> None:
    assert resolve_team("France")[0] == "France"
    assert resolve_team("Brazil")[0] == "Brazil"
    assert resolve_team("USA")[0] == "United States"
    assert resolve_team("South Korea")[0] == "South Korea"
    assert resolve_team("Atlantis") is None


def test_strength_is_monotonic_and_bounded() -> None:
    assert strength_of(1) > strength_of(10) > strength_of(48)
    assert 0.1 <= strength_of(48) <= strength_of(1) <= 0.95


def test_matchup_favours_higher_ranked_home() -> None:
    m = matchup_from_ranking("France vs Brazil")  # France (3) stronger than Brazil (5)
    assert m is not None
    assert m.home.team == "France" and m.away is not None and m.away.team == "Brazil"
    assert m.rel_win_rate > 0.5
    assert m.source == "fifa_ranking"


def test_matchup_is_symmetric_when_reversed() -> None:
    ab = matchup_from_ranking("France vs Brazil")
    ba = matchup_from_ranking("Brazil vs France")
    assert ba.rel_strength < 0.5
    assert abs(ab.rel_strength + ba.rel_strength - 1.0) < 0.01


def test_matchup_single_team_is_absolute() -> None:
    m = matchup_from_ranking("France")
    assert m.away is None
    assert m.rel_strength == m.home.strength


def test_matchup_unknown_home_returns_none() -> None:
    assert matchup_from_ranking("Atlantis vs France") is None


def test_worldcup_agent_integration() -> None:
    ctx = AgentContext(
        entity="France vs Brazil", domain="worldcup", connectors=[WorldCupConnector()]
    )
    m = matchup_from_ranking("France vs Brazil")
    # The three wired agents use real FIFA-ranking signals (no source-specific
    # KeyError -> would otherwise fall back to the neutral error result).
    for name, expected in (
        ("historical", m.rel_win_rate),
        ("trend", m.rel_momentum),
        ("market", m.rel_strength),
    ):
        result = registry.create(name).run(ctx)
        assert result.extra.get("source") == "fifa_ranking", name
        assert result.score == expected, name
    # WC-7: Risk now contributes a real uncertainty signal (no longer abstains).
    risk = registry.create("risk").run(ctx)
    assert risk.extra.get("source") == "fifa_ranking"
    assert risk.extra.get("risk_level") in {"low", "medium", "high"}
    assert risk.extra.get("abstained") is None
    # Contextual still abstains here: neither France nor Brazil is a host nation.
    contextual = registry.create("contextual").run(ctx)
    assert contextual.score == 0.5
    assert contextual.extra.get("abstained") is True


def test_contextual_applies_host_advantage() -> None:
    ctx = AgentContext(
        entity="Mexico vs South Africa", domain="worldcup", connectors=[WorldCupConnector()]
    )
    result = registry.create("contextual").run(ctx)
    assert result.extra.get("source") == "host_advantage"
    assert result.extra.get("host") == "Mexico"
    assert result.score > 0.5  # the host nation is favoured


# --------------------------------------------------------------------------- #
# WC-1b: live form enrichment from football-data.org (offline, with fixtures)
# --------------------------------------------------------------------------- #
def test_wc_form_from_matches() -> None:
    assert wc_form_from_matches(WC_MATCHES, "Argentina") == ["W", "D", "L"]
    assert wc_form_from_matches(WC_MATCHES, "Saudi Arabia") == []  # only a TIMED match
    assert wc_form_from_matches([], "Argentina") == []


def test_enrich_blends_live_form() -> None:
    base = matchup_from_ranking("Argentina vs Saudi Arabia")
    enriched = enrich_matchup_with_form(base, WC_MATCHES)
    assert enriched.home.played == 3  # Argentina W/D/L
    assert enriched.home.win_rate == round(1 / 3, 3)
    assert enriched.home.strength == base.home.strength  # strength stays FIFA-based
    assert enriched.source == "football-data.org"


def test_enrich_pre_tournament_keeps_fifa() -> None:
    base = matchup_from_ranking("Argentina vs Saudi Arabia")
    enriched = enrich_matchup_with_form(base, [])  # no matches played yet
    assert enriched.source == "fifa_ranking"
    assert enriched.home.win_rate == base.home.win_rate
