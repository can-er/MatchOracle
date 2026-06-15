"""Redis cache backend with a transparent in-memory fallback (Sprint 01).

If Redis is unreachable (e.g. tests/CI without a server) the backend degrades to
a process-local dict so the rest of the app keeps working. ``redis`` is imported
lazily, so it isn't a hard dependency unless this backend is selected.
"""

from __future__ import annotations

import json
from typing import Any

from app.logging_config import get_logger

logger = get_logger(__name__)


class RedisCache:
    """Minimal get/set/delete cache over Redis, falling back to memory."""

    def __init__(self, url: str) -> None:
        self._memory: dict[str, str] = {}
        self._client = None
        try:
            import redis

            self._client = redis.Redis.from_url(url, decode_responses=True)
            self._client.ping()
            logger.info("cache.redis.connected", url=url)
        except Exception as exc:  # pragma: no cover - depends on env
            logger.warning("cache.redis.unavailable", error=str(exc))
            self._client = None

    def get(self, key: str) -> Any | None:
        try:
            raw = self._client.get(key) if self._client else self._memory.get(key)
        except Exception:  # pragma: no cover
            raw = self._memory.get(key)
        return json.loads(raw) if raw is not None else None

    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        raw = json.dumps(value, default=str)
        try:
            if self._client:
                self._client.set(key, raw, ex=ttl)
            else:
                self._memory[key] = raw
        except Exception:  # pragma: no cover
            self._memory[key] = raw

    def delete(self, key: str) -> None:
        try:
            if self._client:
                self._client.delete(key)
            else:
                self._memory.pop(key, None)
        except Exception:  # pragma: no cover
            self._memory.pop(key, None)


# Backwards-compatible alias (the class used to be named ``Cache``).
Cache = RedisCache
