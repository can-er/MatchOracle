"""In-process cache with TTL (phase 2).

Used as the default test/CI backend and as a dependency-free fallback. Not shared
across processes, so it's unsuitable for serverless durability - use the Postgres
backend there.
"""

from __future__ import annotations

import time
from typing import Any


class MemoryCache:
    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float | None]] = {}

    def get(self, key: str) -> Any | None:
        item = self._store.get(key)
        if item is None:
            return None
        value, expires_at = item
        if expires_at is not None and expires_at < time.time():
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        self._store[key] = (value, time.time() + ttl if ttl else None)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)
