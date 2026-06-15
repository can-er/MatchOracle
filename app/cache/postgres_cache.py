"""Durable Postgres-backed cache (phase 2).

Persists cache entries in the ``kv_store`` table so process-global state survives
across stateless serverless invocations - the serverless replacement for Redis.
Each operation uses a short-lived session; ``expires_at`` is epoch seconds.
"""

from __future__ import annotations

import time
from typing import Any


class PostgresCache:
    def get(self, key: str) -> Any | None:
        from app.db.base import SessionLocal
        from app.db.models import KeyValue

        with SessionLocal() as session:
            row = session.get(KeyValue, key)
            if row is None:
                return None
            if row.expires_at is not None and row.expires_at < time.time():
                return None
            return row.value

    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        from app.db.base import SessionLocal
        from app.db.models import KeyValue

        expires_at = time.time() + ttl if ttl else None
        with SessionLocal() as session:
            row = session.get(KeyValue, key)
            if row is None:
                session.add(KeyValue(key=key, value=value, expires_at=expires_at))
            else:
                row.value = value
                row.expires_at = expires_at
            session.commit()

    def delete(self, key: str) -> None:
        from app.db.base import SessionLocal
        from app.db.models import KeyValue

        with SessionLocal() as session:
            row = session.get(KeyValue, key)
            if row is not None:
                session.delete(row)
                session.commit()
