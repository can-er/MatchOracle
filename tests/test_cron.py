"""Tests for the daily Vercel cron body (phase 5) - fully offline."""

from __future__ import annotations

import pytest

from app.tasks import mpp_cron
from app.tasks.mpp_cron import run_cron_tick

# Fixtures use clubIds that ARE present in scripts/mpp_club_map.json:
#   367 Czechia vs 522 South Africa, and 614 Brazil vs 1327 Haiti.
_MATCHES = {
    "m1": {
        "date": "2026-06-20T18:00:00Z",
        "home": {"clubId": "mpp_championship_club_367"},  # Czechia
        "away": {"clubId": "mpp_championship_club_522"},  # South Africa
        "quotations": {},
    },
    "m2": {
        "date": "2026-06-20T21:00:00Z",
        "home": {"clubId": "mpp_championship_club_614"},  # Brazil
        "away": {"clubId": "mpp_championship_club_1327"},  # Haiti
        "quotations": {},
    },
}


class _FakeMppClient:
    def __init__(self) -> None:
        self.submitted: list[dict] | None = None

    def next_game_weeks(self, championship_id: int) -> dict:
        return {
            "nextGameWeeks": [
                {
                    "gameWeekNumber": 7,
                    "startDate": "2026-06-20T18:00:00Z",
                    "matchesIds": ["m1", "m2"],
                }
            ]
        }

    def match(self, match_id: str) -> dict:
        return _MATCHES[match_id]

    def submit_forecasts(self, forecasts: list[dict], scope: str = "general") -> list[dict]:
        self.submitted = forecasts
        return [{"match_id": f["match_id"], "ok": True} for f in forecasts]


def test_run_cron_tick_submits_resolved_forecasts(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeMppClient()
    monkeypatch.setattr(mpp_cron, "get_mpp_client", lambda: fake)
    monkeypatch.setattr(mpp_cron, "refresh_worldcup", lambda session: {})
    monkeypatch.setattr("app.config.settings.mpp_enabled", True)

    summary = run_cron_tick(session=None)

    assert summary["refreshed"] is True
    assert summary["game_week"] == 7
    assert summary["submitted"] == 2
    assert summary["skipped"] == 0
    assert summary["errors"] == []

    # The fake recorded exactly two forecasts with integer scores.
    assert fake.submitted is not None
    assert len(fake.submitted) == 2
    assert {f["match_id"] for f in fake.submitted} == {"m1", "m2"}
    for f in fake.submitted:
        assert isinstance(f["home_score"], int)
        assert isinstance(f["away_score"], int)


def test_run_cron_tick_disabled_does_not_call_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> object:
        raise AssertionError("get_mpp_client must not be called when MPP is disabled")

    monkeypatch.setattr(mpp_cron, "get_mpp_client", _boom)
    monkeypatch.setattr(mpp_cron, "refresh_worldcup", lambda session: {})
    monkeypatch.setattr("app.config.settings.mpp_enabled", False)

    summary = run_cron_tick(session=None)

    assert summary["submitted"] == 0
    assert summary["game_week"] is None
    assert summary["skipped"] == 0
    assert summary["errors"] == []
