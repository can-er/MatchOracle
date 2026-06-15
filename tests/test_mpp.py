"""Offline tests for the Mon Petit Prono (MPP) submission package (phase 4).

Fully hermetic: no real network. The auth token POST is monkeypatched and the
MppClient runs against a fake ``httpx.Client`` so we assert exact HTTP paths/bodies
without sockets. The cache is forced to the in-process ``memory`` backend.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.cache import get_cache
from app.mpp import AuthError, MppApiError, MppClient  # noqa: F401
from app.mpp import auth as mpp_auth


@pytest.fixture(autouse=True)
def _memory_cache(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Force the in-process memory cache and start each test with it empty."""
    monkeypatch.setattr("app.config.settings.cache_backend", "memory")
    monkeypatch.setattr("app.config.settings.mpp_refresh_token", "seed-refresh")
    get_cache.cache_clear()
    yield
    get_cache.cache_clear()


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = str(self._payload)

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error", request=None, response=None  # type: ignore[arg-type]
            )


class _FakeHttpClient:
    """Records requests and replays a queued list of responses."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = responses
        self.calls: list[dict] = []

    def request(self, method: str, path: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"method": method, "path": path, **kwargs})
        return self._responses.pop(0)

    def close(self) -> None:  # pragma: no cover - trivial
        pass


def _patch_token_post(
    monkeypatch: pytest.MonkeyPatch, payload: dict
) -> list[dict]:
    """Patch the auth token POST; return a list capturing each POST body."""
    captured: list[dict] = []

    def _fake_post(url: str, json: dict, timeout: float = 0.0) -> _FakeResponse:
        captured.append({"url": url, "json": json})
        return _FakeResponse(200, payload)

    monkeypatch.setattr(mpp_auth.httpx, "post", _fake_post)
    return captured


def test_refresh_persists_rotated_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rotated refresh_token from Auth0 is written back into the cache."""
    captured = _patch_token_post(
        monkeypatch,
        {
            "access_token": "access-1",
            "expires_in": 3600,
            "token_type": "Bearer",
            "refresh_token": "rotated-refresh",
        },
    )

    token = mpp_auth.get_access_token()
    assert token == "access-1"

    # The grant used the seed refresh token.
    assert captured[0]["json"]["refresh_token"] == "seed-refresh"
    assert captured[0]["json"]["grant_type"] == "refresh_token"

    # The cache now holds the ROTATED refresh token, not the seed.
    stored = get_cache().get(mpp_auth.TOKEN_CACHE_KEY)
    assert stored["refresh_token"] == "rotated-refresh"
    assert stored["access_token"] == "access-1"
    assert stored["expires_at"] > 0

    # A second call within validity does NOT refresh again (no new POST).
    assert mpp_auth.get_access_token() == "access-1"
    assert len(captured) == 1


def test_no_refresh_token_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no seed and an empty cache, get_access_token raises AuthError."""
    monkeypatch.setattr("app.config.settings.mpp_refresh_token", "")
    get_cache.cache_clear()
    with pytest.raises(AuthError):
        mpp_auth.get_access_token()


def test_submit_forecast_exact_path_and_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """submit_forecast PATCHes the exact path with the exact JSON body."""
    _patch_token_post(
        monkeypatch,
        {"access_token": "access-1", "expires_in": 3600, "token_type": "Bearer"},
    )
    fake = _FakeHttpClient(
        [_FakeResponse(200, {"general": {"homeScore": 2, "awayScore": 1}})]
    )
    client = MppClient(client=fake)  # type: ignore[arg-type]

    result = client.submit_forecast("mpp_championship_match_123", 2, 1)

    assert result == {"general": {"homeScore": 2, "awayScore": 1}}
    call = fake.calls[0]
    assert call["method"] == "PATCH"
    assert (
        call["path"]
        == "/user-match-forecasts/entity/general/match/mpp_championship_match_123"
    )
    assert call["json"] == {"homeScore": 2, "awayScore": 1, "originPage": "home"}
    # The Bearer header is attached.
    assert call["headers"]["Authorization"] == "Bearer access-1"


def test_submit_forecasts_continues_past_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing item does not stop the rest; per-item ok flags are reported."""
    _patch_token_post(
        monkeypatch,
        {"access_token": "access-1", "expires_in": 3600, "token_type": "Bearer"},
    )
    # First submit OK; second fails with HTTP 500 (twice -> retry also 500);
    # third OK. The 401-retry path is not exercised here.
    fake = _FakeHttpClient(
        [
            _FakeResponse(200, {"general": {"homeScore": 1, "awayScore": 0}}),
            _FakeResponse(500, {"error": "boom"}),
            _FakeResponse(200, {"general": {"homeScore": 0, "awayScore": 0}}),
        ]
    )
    client = MppClient(client=fake)  # type: ignore[arg-type]

    results = client.submit_forecasts(
        [
            {"match_id": "mpp_championship_match_1", "home_score": 1, "away_score": 0},
            {"match_id": "mpp_championship_match_2", "home_score": 3, "away_score": 3},
            {"match_id": "mpp_championship_match_3", "home_score": 0, "away_score": 0},
        ]
    )

    assert [r["ok"] for r in results] == [True, False, True]
    assert results[0]["match_id"] == "mpp_championship_match_1"
    assert results[0]["result"] == {"general": {"homeScore": 1, "awayScore": 0}}
    assert "error" in results[1]
    assert results[2]["ok"] is True


def test_request_retries_once_on_401(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 401 triggers exactly one forced-refresh retry, then succeeds."""
    refresh_calls = {"count": 0}
    real_get = mpp_auth.get_access_token

    def _counting(force_refresh: bool = False) -> str:
        refresh_calls["count"] += 1
        return real_get(force_refresh=force_refresh)

    monkeypatch.setattr("app.mpp.client.get_access_token", _counting)
    _patch_token_post(
        monkeypatch,
        {"access_token": "access-1", "expires_in": 3600, "token_type": "Bearer"},
    )

    fake = _FakeHttpClient(
        [_FakeResponse(401), _FakeResponse(200, {"nextGameWeeks": []})]
    )
    client = MppClient(client=fake)  # type: ignore[arg-type]

    out = client.next_game_weeks(8)
    assert out == {"nextGameWeeks": []}
    assert len(fake.calls) == 2  # initial + one retry
    assert refresh_calls["count"] == 2
