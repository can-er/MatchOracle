"""Bridge from agents to data connectors (Sprint 08).

Agents call :func:`football_matchup` to fetch real head-to-head signals when a
capable connector is wired into the context; it returns ``None`` (→ heuristic
fallback) when no connector is present, the team can't be resolved, or the
source fails.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.agents.base import AgentContext

if TYPE_CHECKING:  # pragma: no cover
    from app.connectors.openligadb import MatchupMetrics


def football_matchup(ctx: AgentContext) -> MatchupMetrics | None:
    """Real head-to-head metrics for ``ctx.entity`` via any connector exposing
    ``matchup_metrics``; ``None`` if unavailable (callers fall back to heuristics)."""
    for connector in ctx.connectors:
        matchup_metrics = getattr(connector, "matchup_metrics", None)
        if callable(matchup_metrics):
            try:
                return matchup_metrics(ctx.entity)
            except Exception:  # never let a flaky source break an agent
                return None
    return None


# --------------------------------------------------------------------------- #
# WC-7: real auxiliary signals for the Contextual & Risk agents (derived from the
# data we already ingest — no extra API keys). News/odds/injury feeds remain a
# future plug-in.
# --------------------------------------------------------------------------- #
HOST_NATIONS = frozenset({"United States", "Canada", "Mexico"})
HOST_EDGE = 0.12  # home-tournament advantage for a host nation


def host_advantage(matchup: MatchupMetrics | None) -> tuple[float, str] | None:
    """Contextual edge for a World Cup 2026 host nation.

    Returns ``(score, host_team)`` favouring the host (score >0.5 if the home/first
    team hosts, <0.5 if the away team does), or ``None`` when neither — or both —
    teams are hosts.
    """
    if matchup is None or matchup.away is None:
        return None
    home, away = matchup.home.team, matchup.away.team
    home_host, away_host = home in HOST_NATIONS, away in HOST_NATIONS
    if home_host == away_host:  # both or neither -> no net host context
        return None
    return (0.5 + HOST_EDGE, home) if home_host else (0.5 - HOST_EDGE, away)


def outcome_uncertainty(matchup: MatchupMetrics | None) -> tuple[float, dict[str, float]] | None:
    """Real match volatility from the score model: ``(uncertainty, 1X2 probs)``.

    Uncertainty is ``1 - max(P)`` — near 0 for a lopsided match, up to ~0.66 for a
    true three-way coin flip.
    """
    if matchup is None or matchup.away is None:
        return None
    from app.prediction.score import predict_scoreline

    sp = predict_scoreline(matchup.home.strength, matchup.away.strength, neutral=True)
    probs = {"home": sp.p_home_win, "draw": sp.p_draw, "away": sp.p_away_win}
    return round(1.0 - max(probs.values()), 3), probs
