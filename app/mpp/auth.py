"""Auth0 refresh-token auth for Mon Petit Prono (phase 4).

The MPP API authenticates via an OAuth2 refresh-token grant against Auth0. Token
rotation is ON: every refresh may return a *new* refresh_token that invalidates
the previous one, so the new token must be persisted durably.

Serverless has no writable filesystem, so the token store lives in the app cache
(see :func:`app.cache.get_cache`) under the key ``mpp:tokens``. The stored value
is a dict::

    {"access_token": str, "expires_at": float (epoch seconds), "refresh_token": str}

The entry is written with ``ttl=0`` so the cache never evicts it. On first use the
store is seeded from ``settings.mpp_refresh_token``. A refresh fires when the
access token is within 90 seconds of expiry (or ``force_refresh`` is set), after
which the (possibly rotated) refresh token, new access token and freshly computed
expiry are written back.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from app.cache import get_cache
from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

#: Cache key for the durable, never-expiring token store.
TOKEN_CACHE_KEY = "mpp:tokens"
#: Refresh this many seconds before the access token actually expires.
REFRESH_SKEW_SECONDS = 90


class AuthError(RuntimeError):
    """Raised when MPP authentication cannot proceed (e.g. no refresh token)."""


def _load_tokens() -> dict[str, Any]:
    """Load the token store, seeding the refresh token from settings on first use."""
    cache = get_cache()
    tokens = cache.get(TOKEN_CACHE_KEY)
    if not isinstance(tokens, dict):
        tokens = {}
    if not tokens.get("refresh_token"):
        seed = settings.mpp_refresh_token
        if seed:
            tokens = {
                "access_token": tokens.get("access_token", ""),
                "expires_at": float(tokens.get("expires_at", 0.0)),
                "refresh_token": seed,
            }
    return tokens


def _store_tokens(tokens: dict[str, Any]) -> None:
    """Persist the token store durably (ttl=0 -> never evicted)."""
    get_cache().set(TOKEN_CACHE_KEY, tokens, ttl=0)


def _refresh(refresh_token: str) -> dict[str, Any]:
    """Exchange a refresh token for a new access token via the Auth0 grant."""
    url = f"{settings.mpp_auth0_domain}/oauth/token"
    payload = {
        "grant_type": "refresh_token",
        "client_id": settings.mpp_auth0_client_id,
        "refresh_token": refresh_token,
    }
    resp = httpx.post(url, json=payload, timeout=15.0)
    resp.raise_for_status()
    data = resp.json()

    access_token = data.get("access_token")
    if not access_token:
        raise AuthError("Auth0 refresh response did not include an access_token.")

    expires_in = int(data.get("expires_in", 0))
    # Rotation: keep the new refresh token if the server issued one, else reuse.
    new_refresh = data.get("refresh_token") or refresh_token
    tokens = {
        "access_token": access_token,
        "expires_at": time.time() + expires_in,
        "refresh_token": new_refresh,
    }
    _store_tokens(tokens)
    if new_refresh != refresh_token:
        logger.info("mpp.auth.refresh_token_rotated")
    return tokens


def get_access_token(force_refresh: bool = False) -> str:
    """Return a valid MPP access token, refreshing (and persisting) as needed.

    Raises :class:`AuthError` when there is no refresh token at all (nothing in the
    cache and ``settings.mpp_refresh_token`` is empty).
    """
    tokens = _load_tokens()
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        raise AuthError(
            "No MPP refresh token available. Seed settings.mpp_refresh_token "
            "(MO_MPP_REFRESH_TOKEN) once so rotation can take over."
        )

    access_token = tokens.get("access_token")
    expires_at = float(tokens.get("expires_at", 0.0))
    needs_refresh = (
        force_refresh
        or not access_token
        or time.time() >= expires_at - REFRESH_SKEW_SECONDS
    )
    if needs_refresh:
        tokens = _refresh(refresh_token)
        access_token = tokens["access_token"]
    return access_token
