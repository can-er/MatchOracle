"""Typed Mon Petit Prono (MPP) REST client (phase 4).

Reads game weeks / matches and submits forecasts. Every request carries a Bearer
access token from :func:`app.mpp.auth.get_access_token`; on an HTTP 401 the client
retries exactly once with ``force_refresh=True`` to ride out a rotated/expired
access token.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.config import settings
from app.logging_config import get_logger
from app.mpp.auth import get_access_token

logger = get_logger(__name__)

#: Browser-ish identity the MPP edge expects on API calls.
_USER_AGENT = "MatchOracle/1.0 (+https://github.com/matchoracle)"

_DEFAULT_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Origin": "https://mpp.football",
    "Referer": "https://mpp.football/",
    "User-Agent": _USER_AGENT,
}


class MppApiError(RuntimeError):
    """Raised when an MPP API call fails for a non-authentication reason."""


class MppClient:
    """Typed client for the MPP REST API.

    Args:
        client: optional pre-built ``httpx.Client`` (used for tests / injection).
            When omitted, one is built from ``settings.mpp_api_base`` with the
            default MPP headers.
    """

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(
            base_url=settings.mpp_api_base.rstrip("/"),
            headers=dict(_DEFAULT_HEADERS),
            timeout=15.0,
        )

    # ----------------------------------------------------------------- core --- #
    def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        """Issue an authenticated request, retrying once on 401 with a refresh."""
        for attempt in range(2):
            force = attempt == 1
            token = get_access_token(force_refresh=force)
            headers = {"Authorization": f"Bearer {token}"}
            resp = self._client.request(method, path, headers=headers, **kwargs)
            if resp.status_code == 401 and attempt == 0:
                logger.info("mpp.client.unauthorized_retry", path=path)
                continue
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise MppApiError(
                    f"MPP {method} {path} failed: {resp.status_code} {resp.text}"
                ) from exc
            return resp.json()
        # Unreachable: the loop either returns or raises, but keep the type checker
        # and any future edits honest.
        raise MppApiError(f"MPP {method} {path} exhausted retries.")

    # ----------------------------------------------------------------- reads --- #
    def next_game_weeks(self, championship_id: int) -> dict:
        """Return the upcoming game weeks for a championship.

        Shape: ``{"nextGameWeeks": [{"gameWeekNumber", "startDate", "endDate",
        "startIn", "matchesIds": [...]}, ...]}``.
        """
        return self._request(
            "GET", f"/championship-calendar/{championship_id}/next-game-weeks"
        )

    def match(self, match_id: str) -> dict:
        """Return a single match's detail (quotations, clubs, date, ...)."""
        return self._request("GET", f"/championship-match/{match_id}")

    # ----------------------------------------------------------- submissions --- #
    def submit_forecast(
        self,
        match_id: str,
        home_score: int,
        away_score: int,
        scope: str = "general",
    ) -> dict:
        """Submit (PATCH) a single score forecast for ``match_id``.

        ``match_id`` is the full forecast id, e.g.
        ``"mpp_championship_match_<numericId>"``. Returns the API response, e.g.
        ``{"general": {"homeScore", "awayScore", ...}}``.
        """
        body = {
            "homeScore": int(home_score),
            "awayScore": int(away_score),
            "originPage": "home",
        }
        return self._request(
            "PATCH",
            f"/user-match-forecasts/entity/{scope}/match/{match_id}",
            json=body,
        )

    def submit_forecasts(
        self, forecasts: list[dict], scope: str = "general"
    ) -> list[dict]:
        """Submit many forecasts; one failure never aborts the rest.

        Each input dict needs ``match_id``, ``home_score`` and ``away_score``.
        Returns one result per input::

            {"match_id": str, "ok": bool, "result": dict}   # on success
            {"match_id": str, "ok": bool, "error": str}     # on failure
        """
        results: list[dict] = []
        for forecast in forecasts:
            match_id = forecast.get("match_id")
            try:
                result = self.submit_forecast(
                    match_id,
                    forecast["home_score"],
                    forecast["away_score"],
                    scope=scope,
                )
                results.append({"match_id": match_id, "ok": True, "result": result})
            except Exception as exc:  # noqa: BLE001 - isolate per-item failures
                logger.warning(
                    "mpp.client.submit_failed", match_id=match_id, error=str(exc)
                )
                results.append({"match_id": match_id, "ok": False, "error": str(exc)})
        return results

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()


def get_mpp_client() -> MppClient:
    """Build a ready-to-use :class:`MppClient` from settings."""
    return MppClient()
