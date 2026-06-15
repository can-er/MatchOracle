"""Shared test fixtures — keep the whole suite hermetic (offline).

A real `.env` may carry a football-data.org key, the OpenLigaDB connector fetches
with no key, and a running Redis may hold real cached data — any of which could
make a test hit the network or read live data. We blank the key, point the cache
at an unreachable Redis (so it falls back to an empty in-memory cache), and
disable outbound HTTP. Connectors then fall back to their offline path.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

from app.cache import get_cache


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr("app.config.settings.football_data_api_key", "")
    # Unreachable Redis -> Cache falls back to a fresh, empty in-memory dict.
    monkeypatch.setattr("app.config.settings.redis_url", "redis://127.0.0.1:1")
    get_cache.cache_clear()

    def _no_network(*args: object, **kwargs: object) -> object:
        raise RuntimeError("outbound HTTP is disabled during tests")

    monkeypatch.setattr(httpx, "get", _no_network)
    yield
    get_cache.cache_clear()
