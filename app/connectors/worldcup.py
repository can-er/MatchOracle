"""World Cup 2026 connector (Sprint WC-1).

Predicts national-team matchups for the FIFA World Cup 2026. Before the
tournament kicks off there are no results/standings yet, so the real
pre-tournament signal is the **FIFA ranking** (team strength) — embedded here as
a static seed so it works with no API key. Once matches are played, live form
and results from football-data.org (competition ``WC``, key in ``.env``) will
enrich the signals — wired in WC-1b.

It returns the same :class:`~app.connectors.openligadb.MatchupMetrics` the agents
already consume, so the head-to-head machinery is reused unchanged.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.cache import get_cache
from app.config import settings
from app.connectors.base import BaseConnector
from app.connectors.openligadb import (
    MatchupMetrics,
    TeamMetrics,
    _normalise,
    _relative,
    momentum_from_form,
    opponent_team,
    subject_team,
)
from app.logging_config import get_logger

logger = get_logger(__name__)

# Strength seed for the 48 teams that actually qualified for World Cup 2026 (team
# names exactly as football-data.org returns them). The value is an *approximate*
# FIFA-style ranking (rank 1 = strongest), ordered by strength from early-2026
# knowledge — not the official points table. WC-6 calibration and live form refine
# it; this just gives every qualified team a sensible pre-tournament strength
# (no more neutral 0.5 fallbacks).
FIFA_RANKING: dict[str, int] = {
    "Argentina": 1,
    "Spain": 2,
    "France": 3,
    "England": 4,
    "Brazil": 5,
    "Portugal": 6,
    "Netherlands": 7,
    "Belgium": 8,
    "Croatia": 9,
    "Germany": 10,
    "Morocco": 11,
    "Colombia": 12,
    "Uruguay": 13,
    "United States": 14,
    "Mexico": 15,
    "Switzerland": 16,
    "Japan": 17,
    "Senegal": 18,
    "Iran": 19,
    "South Korea": 20,
    "Ecuador": 21,
    "Austria": 22,
    "Australia": 23,
    "Sweden": 24,
    "Norway": 25,
    "Turkey": 26,
    "Egypt": 27,
    "Ivory Coast": 28,
    "Algeria": 29,
    "Scotland": 30,
    "Paraguay": 31,
    "Tunisia": 32,
    "Canada": 33,
    "Panama": 34,
    "Qatar": 35,
    "Czechia": 36,
    "Congo DR": 37,
    "Saudi Arabia": 38,
    "Uzbekistan": 39,
    "Iraq": 40,
    "South Africa": 41,
    "Ghana": 42,
    "Jordan": 43,
    "Cape Verde Islands": 44,
    "Bosnia-Herzegovina": 45,
    "Haiti": 46,
    "Curaçao": 47,
    "New Zealand": 48,
}

# Common aliases so user-friendly names resolve to the ranking keys.
_ALIASES: dict[str, str] = {
    "usa": "United States",
    "unitedstates": "United States",
    "us": "United States",
    "korea": "South Korea",
    "korearepublic": "South Korea",
    "ivorycoast": "Ivory Coast",
    "cotedivoire": "Ivory Coast",
    "holland": "Netherlands",
    "thenetherlands": "Netherlands",
    "turkiye": "Turkey",
    "capeverde": "Cape Verde Islands",
    "bosnia": "Bosnia-Herzegovina",
    "drcongo": "Congo DR",
    "drc": "Congo DR",
    "czechrepublic": "Czechia",
}

_SEED_SPAN = 50  # controls how strength decays with rank


def strength_of(rank: int) -> float:
    """Map a FIFA rank to a strength score in [0.1, 0.95] (rank 1 = strongest)."""
    return round(max(0.1, min(0.95, 1.0 - (rank - 1) / _SEED_SPAN)), 3)


def resolve_team(name: str) -> tuple[str, int] | None:
    """Resolve a national-team name to ``(canonical, rank)`` via the seed/aliases."""
    query = _normalise(name)
    if not query:
        return None
    if query in _ALIASES:
        team = _ALIASES[query]
        return team, FIFA_RANKING[team]
    for team, rank in FIFA_RANKING.items():
        canonical = _normalise(team)
        if query in canonical or canonical in query:
            return team, rank
    return None


def team_strength(name: str) -> float:
    """Strength in [0.1, 0.95] from the FIFA seed; 0.5 for unseeded teams."""
    resolved = resolve_team(name)
    return strength_of(resolved[1]) if resolved else 0.5


def normalise_group(group: str) -> str:
    """Normalise a group label to football-data.org's form: 'A' -> 'GROUP_A'."""
    s = group.upper().replace(" ", "_").strip()
    if not s:
        return ""
    if s.startswith("GROUP_"):
        return s
    if s.startswith("GROUP"):
        return "GROUP_" + s[len("GROUP") :].lstrip("_")
    return f"GROUP_{s}"


def _team_metrics(team: str, rank: int) -> TeamMetrics:
    # Pre-tournament: all signals derive from FIFA strength (no matches yet).
    s = strength_of(rank)
    return TeamMetrics(
        team=team,
        rank=rank,
        played=0,
        win_rate=s,
        strength=s,
        momentum=s,
        form="",
        detail={"fifa_rank": rank, "source": "fifa_ranking"},
    )


def matchup_from_ranking(entity: str) -> MatchupMetrics | None:
    """Head-to-head metrics from the FIFA-ranking seed (home perspective)."""
    home_resolved = resolve_team(subject_team(entity))
    if home_resolved is None:
        return None
    home = _team_metrics(*home_resolved)

    opponent = opponent_team(entity)
    away_resolved = resolve_team(opponent) if opponent else None
    away = _team_metrics(*away_resolved) if away_resolved else None

    if away is not None:
        rel_win = _relative(home.win_rate, away.win_rate)
        rel_str = _relative(home.strength, away.strength)
        rel_mom = (
            _relative(home.momentum, away.momentum) if home.momentum and away.momentum else None
        )
        label = f"{home.team} vs {away.team}"
    else:
        rel_win, rel_str, rel_mom = home.win_rate, home.strength, home.momentum
        label = home.team

    return MatchupMetrics(
        home=home,
        away=away,
        rel_win_rate=rel_win,
        rel_strength=rel_str,
        rel_momentum=rel_mom,
        label=label,
        source="fifa_ranking",
    )


# --------------------------------------------------------------------------- #
# Live enrichment from football-data.org (WC-1b) — pure functions
# --------------------------------------------------------------------------- #
def wc_form_from_matches(matches: list[dict], team: str) -> list[str]:
    """Chronological W/D/L for a team's FINISHED World Cup matches."""
    norm = _normalise(team)
    dated: list[tuple[str, str]] = []
    for m in matches:
        if m.get("status") != "FINISHED":
            continue
        home = _normalise((m.get("homeTeam") or {}).get("name") or "")
        away = _normalise((m.get("awayTeam") or {}).get("name") or "")
        is_home = bool(norm) and (norm in home or home in norm)
        is_away = bool(norm) and (norm in away or away in norm)
        if not (is_home or is_away):
            continue
        full_time = (m.get("score") or {}).get("fullTime") or {}
        gh, ga = full_time.get("home"), full_time.get("away")
        if gh is None or ga is None:
            continue
        gf, against = (gh, ga) if is_home else (ga, gh)
        dated.append(
            (m.get("utcDate") or "", "W" if gf > against else "D" if gf == against else "L")
        )
    dated.sort(key=lambda x: x[0])
    return [result for _, result in dated]


def _enrich_team(team: TeamMetrics, matches: list[dict]) -> TeamMetrics:
    form = wc_form_from_matches(matches, team.team)
    if not form:
        return team  # no tournament matches played yet -> keep FIFA-based metrics
    wins = form.count("W")
    return TeamMetrics(
        team=team.team,
        rank=team.rank,
        played=len(form),
        win_rate=round(wins / len(form), 3),
        strength=team.strength,  # strength stays FIFA-based
        momentum=momentum_from_form(form[-5:]),
        form="".join(form[-5:]),
        detail={**team.detail, "wc_played": len(form), "source": "football-data.org"},
    )


def enrich_matchup_with_form(matchup: MatchupMetrics, matches: list[dict]) -> MatchupMetrics:
    """Blend live World Cup form (finished matches) into a FIFA-seed matchup."""
    home = _enrich_team(matchup.home, matches)
    away = _enrich_team(matchup.away, matches) if matchup.away else None
    used_live = home.detail.get("source") == "football-data.org" or (
        away is not None and away.detail.get("source") == "football-data.org"
    )

    if away is not None:
        rel_win = _relative(home.win_rate, away.win_rate)
        rel_str = _relative(home.strength, away.strength)
        rel_mom = (
            _relative(home.momentum, away.momentum) if home.momentum and away.momentum else None
        )
    else:
        rel_win, rel_str, rel_mom = home.win_rate, home.strength, home.momentum

    return MatchupMetrics(
        home=home,
        away=away,
        rel_win_rate=rel_win,
        rel_strength=rel_str,
        rel_momentum=rel_mom,
        label=matchup.label,
        source="football-data.org" if used_live else "fifa_ranking",
    )


class WorldCupConnector(BaseConnector):
    name = "worldcup"
    type = "rest"
    domain = "worldcup"

    def __init__(self) -> None:
        self.api_key = settings.football_data_api_key
        self.base_url = settings.football_data_base_url.rstrip("/")
        self.competition = settings.wc_competition

    def _get(self, path: str) -> Any:
        cache = get_cache()
        key = f"fdo:{path}"
        cached = cache.get(key)
        if cached is not None:
            return cached
        resp = httpx.get(
            f"{self.base_url}/{path}",
            headers={"X-Auth-Token": self.api_key},
            timeout=8.0,
        )
        resp.raise_for_status()
        data = resp.json()
        cache.set(key, data, ttl=600)  # refresh form/results every ~10 min
        return data

    def _wc_matches(self) -> list[dict]:
        try:
            data = self._get(f"competitions/{self.competition}/matches")
            return list(data.get("matches") or [])
        except Exception as exc:
            logger.warning("connector.worldcup.fetch_failed", error=str(exc))
            return []

    def matchday_fixtures(self, matchday: int) -> list[dict] | None:
        """Group-stage matches for a given matchday (None if no data)."""
        matches = self._wc_matches()
        if not matches:
            return None
        return [m for m in matches if m.get("matchday") == matchday]

    def finished_results(self) -> dict[tuple[str, str], str]:
        """Real finished results keyed by ``(home, away)`` -> ``"h-a"`` scoreline.

        Powers the autonomous feedback loop: no human input, just the real
        outcomes football-data.org reports as matches finish.
        """
        results: dict[tuple[str, str], str] = {}
        for match in self._wc_matches():
            if match.get("status") != "FINISHED":
                continue
            home = (match.get("homeTeam") or {}).get("name")
            away = (match.get("awayTeam") or {}).get("name")
            full_time = (match.get("score") or {}).get("fullTime") or {}
            h, a = full_time.get("home"), full_time.get("away")
            if home and away and h is not None and a is not None:
                results[(home, away)] = f"{h}-{a}"
        return results

    def group_data(
        self, group: str
    ) -> (
        tuple[str, dict[str, float], list[tuple[str, str]], dict[tuple[str, str], tuple[int, int]]]
        | None
    ):
        """Return (group_label, team->strength, fixtures, played-results) for a group."""
        matches = self._wc_matches()
        if not matches:
            return None
        target = normalise_group(group)
        teams: dict[str, float] = {}
        fixtures: list[tuple[str, str]] = []
        results: dict[tuple[str, str], tuple[int, int]] = {}
        for match in matches:
            if normalise_group(match.get("group") or "") != target:
                continue
            home = (match.get("homeTeam") or {}).get("name")
            away = (match.get("awayTeam") or {}).get("name")
            if not home or not away:
                continue
            teams.setdefault(home, team_strength(home))
            teams.setdefault(away, team_strength(away))
            fixtures.append((home, away))
            if match.get("status") == "FINISHED":
                full_time = (match.get("score") or {}).get("fullTime") or {}
                if full_time.get("home") is not None and full_time.get("away") is not None:
                    results[(home, away)] = (full_time["home"], full_time["away"])
        if not fixtures:
            return None
        return target, teams, fixtures, results

    def all_group_data(
        self,
    ) -> dict[str, tuple[dict[str, float], list[tuple[str, str]], dict]] | None:
        """All groups' (team->strength, fixtures, results), keyed by group label."""
        matches = self._wc_matches()
        if not matches:
            return None
        groups: dict[str, tuple[dict[str, float], list[tuple[str, str]], dict]] = {}
        for match in matches:
            label = normalise_group(match.get("group") or "")
            if not label:  # knockout matches have no group
                continue
            home = (match.get("homeTeam") or {}).get("name")
            away = (match.get("awayTeam") or {}).get("name")
            if not home or not away:
                continue
            teams, fixtures, results = groups.setdefault(label, ({}, [], {}))
            teams.setdefault(home, team_strength(home))
            teams.setdefault(away, team_strength(away))
            fixtures.append((home, away))
            if match.get("status") == "FINISHED":
                full_time = (match.get("score") or {}).get("fullTime") or {}
                if full_time.get("home") is not None and full_time.get("away") is not None:
                    results[(home, away)] = (full_time["home"], full_time["away"])
        return groups or None

    def health(self) -> bool:
        # The FIFA-ranking seed is always available; the live API is optional.
        return True

    def matchup_metrics(self, entity: str) -> MatchupMetrics | None:
        """Head-to-head metrics: FIFA-ranking strength, enriched with live World
        Cup form from football-data.org once the key is set and matches are played."""
        base = matchup_from_ranking(entity)
        if base is None or not self.api_key:
            return base
        matches = self._wc_matches()
        if not matches:
            return base
        return enrich_matchup_with_form(base, matches)
