"""OpenLigaDB football connector (Sprint 08).

Free, key-less, uncapped football data (https://api.openligadb.de). It turns a
team entity into real signals for the agents:

- **win_rate**  — season wins / matches      (Historical agent)
- **strength**  — points per match / 3       (Market agent)
- **momentum**  — recency-weighted last 5     (Trend agent)

The HTTP layer is cached (Redis, with in-memory fallback) so all agents share a
single fetch per (league, season). The computation is split into pure functions
so it can be unit-tested offline with fixture data.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.cache import get_cache
from app.config import settings
from app.connectors.base import BaseConnector
from app.logging_config import get_logger

logger = get_logger(__name__)

_SEPARATORS = (" vs ", " v ", " versus ", " - ", " x ", " contre ")
_RESULT_FINAL = 2  # openligadb resultTypeID for the final score


@dataclass
class TeamMetrics:
    team: str
    rank: int
    played: int
    win_rate: float
    strength: float
    momentum: float | None
    form: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchupMetrics:
    """Head-to-head signals from the first (home) team's perspective.

    Each ``rel_*`` value is in [0,1]: 0.5 = even, >0.5 favours ``home``. When the
    opponent can't be resolved, ``away`` is None and the relative values fall back
    to ``home``'s absolute metrics (single-team behaviour).
    """

    home: TeamMetrics
    away: TeamMetrics | None
    rel_win_rate: float
    rel_strength: float
    rel_momentum: float | None
    label: str
    source: str = "connector"


# --------------------------------------------------------------------------- #
# Pure helpers (no I/O — unit-tested directly)
# --------------------------------------------------------------------------- #
def _normalise(text: str) -> str:
    """Lowercase, strip accents, keep alphanumerics only (robust matching)."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if ch.isalnum())


def subject_team(entity: str) -> str:
    """Extract the subject (home) team: 'Bayern vs Dortmund' -> 'Bayern'."""
    low = entity.lower()
    for sep in _SEPARATORS:
        idx = low.find(sep)
        if idx != -1:
            return entity[:idx].strip()
    return entity.strip()


def opponent_team(entity: str) -> str:
    """Extract the opponent (away) team: 'Bayern vs Dortmund' -> 'Dortmund' (else '')."""
    low = entity.lower()
    for sep in _SEPARATORS:
        idx = low.find(sep)
        if idx != -1:
            return entity[idx + len(sep) :].strip()
    return ""


def _relative(home: float, away: float) -> float:
    """Map two [0,1] values to a home-perspective edge in [0,1] (0.5 = even)."""
    return round(min(1.0, max(0.0, 0.5 + (home - away) / 2)), 3)


def resolve_team(table: list[dict], entity: str) -> tuple[int, dict] | None:
    """Find ``(rank_index, row)`` for the entity in a standings table."""
    subject = _normalise(subject_team(entity))
    if not subject:
        return None
    for i, row in enumerate(table):
        name = _normalise(str(row.get("teamName", "")))
        short = _normalise(str(row.get("shortName", "")))
        if name and (subject in name or name in subject):
            return i, row
        if short and (subject in short or short in subject):
            return i, row
    return None


def _final_score(match: dict) -> tuple[int, int] | None:
    for result in match.get("matchResults") or []:
        if result.get("resultTypeID") == _RESULT_FINAL:
            return int(result.get("pointsTeam1", 0)), int(result.get("pointsTeam2", 0))
    return None


def recent_form(matches: list[dict], team_name: str, n: int = 5) -> list[str]:
    """Last ``n`` results ('W'/'D'/'L') for the team, oldest→newest."""
    finished = [
        m
        for m in matches
        if m.get("matchIsFinished")
        and team_name in (m["team1"]["teamName"], m["team2"]["teamName"])
    ]
    finished.sort(key=lambda m: m.get("matchDateTime") or "")
    out: list[str] = []
    for m in finished[-n:]:
        score = _final_score(m)
        if score is None:
            continue
        home, away = score
        is_home = team_name == m["team1"]["teamName"]
        gf, ga = (home, away) if is_home else (away, home)
        out.append("W" if gf > ga else "D" if gf == ga else "L")
    return out


def momentum_from_form(form: list[str]) -> float | None:
    """Recency-weighted momentum in [0,1] (most recent result weighs most)."""
    if not form:
        return None
    points = {"W": 1.0, "D": 0.5, "L": 0.0}
    weights = [i + 1 for i in range(len(form))]
    total = sum(points[r] * w for r, w in zip(form, weights, strict=True))
    return round(total / sum(weights), 3)


def metrics_from(table: list[dict], matches: list[dict], entity: str) -> TeamMetrics | None:
    """Compute :class:`TeamMetrics` from raw table+matches (pure)."""
    resolved = resolve_team(table, entity)
    if resolved is None:
        return None
    rank_idx, row = resolved
    team = str(row.get("teamName", ""))
    played = int(row.get("matches", 0) or 0)
    won = int(row.get("won", 0) or 0)
    points = int(row.get("points", 0) or 0)
    if played <= 0:
        return None

    win_rate = round(won / played, 3)
    strength = round(min(1.0, points / (played * 3)), 3)
    form = recent_form(matches, team)
    momentum = momentum_from_form(form)

    return TeamMetrics(
        team=team,
        rank=rank_idx + 1,
        played=played,
        win_rate=win_rate,
        strength=strength,
        momentum=momentum,
        form="".join(form),
        detail={
            "points": points,
            "won": won,
            "draw": int(row.get("draw", 0) or 0),
            "lost": int(row.get("lost", 0) or 0),
            "goal_diff": int(row.get("goalDiff", 0) or 0),
        },
    )


def matchup_from(table: list[dict], matches: list[dict], entity: str) -> MatchupMetrics | None:
    """Head-to-head metrics (home perspective). None if the home team is unknown.

    With both teams resolved the signals are relative (home vs away). With only
    the home team resolved they fall back to its absolute metrics.
    """
    home = metrics_from(table, matches, subject_team(entity))
    if home is None:
        return None
    opponent = opponent_team(entity)
    away = metrics_from(table, matches, opponent) if opponent else None

    if away is not None:
        rel_win = _relative(home.win_rate, away.win_rate)
        rel_str = _relative(home.strength, away.strength)
        rel_mom = (
            _relative(home.momentum, away.momentum)
            if home.momentum is not None and away.momentum is not None
            else home.momentum
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
        source="openligadb",
    )


# --------------------------------------------------------------------------- #
# Connector (I/O + cache)
# --------------------------------------------------------------------------- #
class OpenLigaDBConnector(BaseConnector):
    name = "openligadb"
    type = "rest"
    domain = "sports"

    def __init__(
        self,
        league: str | None = None,
        season: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
        cache_ttl: int = 3600,
    ) -> None:
        self.league = league or settings.football_league
        self.season = season or settings.football_season
        self.base_url = (base_url or settings.openligadb_base_url).rstrip("/")
        self.timeout = timeout
        self.cache_ttl = cache_ttl

    def _get(self, path: str) -> Any:
        cache = get_cache()
        key = f"oldb:{self.base_url}:{path}"
        cached = cache.get(key)
        if cached is not None:
            return cached
        resp = httpx.get(f"{self.base_url}/{path}", timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        cache.set(key, data, ttl=self.cache_ttl)
        return data

    def health(self) -> bool:
        try:
            self._get(f"getbltable/{self.league}/{self.season}")
            return True
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("connector.openligadb.unhealthy", error=str(exc))
            return False

    def _season_data(self) -> tuple[list[dict], list[dict]] | None:
        try:
            table = self._get(f"getbltable/{self.league}/{self.season}")
            matches = self._get(f"getmatchdata/{self.league}/{self.season}")
        except Exception as exc:
            logger.warning("connector.openligadb.fetch_failed", error=str(exc))
            return None
        return table, matches

    def team_metrics(self, entity: str) -> TeamMetrics | None:
        """Resolve the entity to a single team and compute its real metrics."""
        data = self._season_data()
        return metrics_from(*data, entity) if data else None

    def matchup_metrics(self, entity: str) -> MatchupMetrics | None:
        """Head-to-head metrics for 'A vs B' (home perspective), or None."""
        data = self._season_data()
        return matchup_from(*data, entity) if data else None
