"""Caching layer (Sprint 01; pluggable backends in phase 2).

``get_cache()`` returns the backend selected by ``settings.cache_backend``:

* ``redis``    - Redis with an in-memory fallback (default, local/Docker),
* ``postgres`` - durable ``kv_store`` table (serverless: Vercel + Supabase),
* ``memory``   - process-local dict (tests / dependency-free).

All backends share the :class:`~app.cache.base.Cache` get/set/delete contract.
"""

from __future__ import annotations

from functools import lru_cache

from app.cache.base import Cache
from app.config import settings


@lru_cache
def get_cache() -> Cache:
    backend = settings.cache_backend.lower()
    if backend == "postgres":
        from app.cache.postgres_cache import PostgresCache

        return PostgresCache()
    if backend == "memory":
        from app.cache.memory_cache import MemoryCache

        return MemoryCache()
    from app.cache.redis_cache import RedisCache

    return RedisCache(settings.redis_url)


__all__ = ["Cache", "get_cache"]
