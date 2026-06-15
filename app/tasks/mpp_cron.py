"""Daily Vercel cron body (phase 5).

One scheduled pass does two things:

1. Refresh World Cup predictions and run the autonomous feedback loop
   (:func:`app.tasks.refresh.refresh_worldcup`), so the predictions stay fresh
   and the agent weights re-tune from the tournament's own results.
2. Submit MatchOracle's forecasts for the next open Mon Petit Prono (MPP) game
   week, when MPP submission is enabled.

Every external call is wrapped defensively so a single bad match (or a failing
refresh) never aborts the whole run.
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import settings
from app.connectors.worldcup import team_strength
from app.logging_config import get_logger
from app.mpp import get_mpp_client
from app.prediction.score import predict_scoreline
from app.tasks.refresh import refresh_worldcup

logger = get_logger(__name__)

_CLUB_MAP = Path(__file__).resolve().parents[2] / "scripts" / "mpp_club_map.json"
_CLUB_PREFIX = "mpp_championship_club_"


def _load_club_map() -> dict[str, str]:
    """Load the clubId -> team-name map once. Returns {} on any failure."""
    try:
        data = json.loads(_CLUB_MAP.read_text(encoding="utf-8"))
        clubs = data.get("clubs", {})
        return {str(k): str(v) for k, v in clubs.items()}
    except Exception as exc:  # never abort the run over a missing/bad map
        logger.warning("mpp.cron.club_map_failed", error=str(exc))
        return {}


def _team_name(club_id: object, club_map: dict[str, str]) -> str | None:
    """Resolve an MPP clubId (e.g. ``mpp_championship_club_367``) to a team name."""
    if not isinstance(club_id, str):
        return None
    key = club_id[len(_CLUB_PREFIX):] if club_id.startswith(_CLUB_PREFIX) else club_id
    return club_map.get(key)


def run_cron_tick(session: Session) -> dict:
    """Refresh predictions + autonomous learning, then submit MPP forecasts."""
    refreshed = False
    try:
        refresh_worldcup(session)
        refreshed = True
    except Exception as exc:  # log + continue: MPP submission still runs
        logger.warning("mpp.cron.refresh_failed", error=str(exc))

    if not settings.mpp_enabled:
        return {
            "refreshed": refreshed,
            "game_week": None,
            "submitted": 0,
            "skipped": 0,
            "errors": [],
        }

    skipped = 0
    errors: list[str] = []
    game_week: int | None = None
    forecasts: list[dict] = []

    try:
        client = get_mpp_client()
        weeks = client.next_game_weeks(settings.mpp_championship_id)
        entries = (weeks or {}).get("nextGameWeeks") or []
        if not entries:
            logger.info("mpp.cron.no_open_game_week")
            return {
                "refreshed": refreshed,
                "game_week": None,
                "submitted": 0,
                "skipped": 0,
                "errors": [],
            }

        week = entries[0]
        game_week = week.get("gameWeekNumber")
        club_map = _load_club_map()

        for match_id in week.get("matchesIds") or []:
            try:
                m = client.match(match_id)
                home = _team_name((m.get("home") or {}).get("clubId"), club_map)
                away = _team_name((m.get("away") or {}).get("clubId"), club_map)
                if home is None or away is None:
                    skipped += 1
                    logger.info("mpp.cron.unmapped_match", match_id=str(match_id))
                    continue
                sp = predict_scoreline(
                    team_strength(home), team_strength(away), neutral=True
                )
                forecasts.append(
                    {
                        "match_id": match_id,
                        "home_score": sp.home_goals,
                        "away_score": sp.away_goals,
                    }
                )
            except Exception as exc:
                errors.append(str(match_id))
                logger.warning(
                    "mpp.cron.match_failed", match_id=str(match_id), error=str(exc)
                )

        submitted = 0
        if forecasts:
            try:
                results = client.submit_forecasts(forecasts, scope="general")
                for r in results or []:
                    if r.get("ok"):
                        submitted += 1
                    else:
                        errors.append(str(r.get("match_id")))
            except Exception as exc:  # whole batch failed
                logger.warning("mpp.cron.submit_failed", error=str(exc))
                errors.extend(str(f["match_id"]) for f in forecasts)
                submitted = 0
    except Exception as exc:  # any MPP-side failure: report what we have
        logger.warning("mpp.cron.failed", error=str(exc))
        return {
            "refreshed": refreshed,
            "game_week": game_week,
            "submitted": 0,
            "skipped": skipped,
            "errors": errors,
        }

    summary = {
        "refreshed": refreshed,
        "game_week": game_week,
        "submitted": submitted,
        "skipped": skipped,
        "errors": errors,
    }
    logger.info("mpp.cron.done", **summary)
    return summary
