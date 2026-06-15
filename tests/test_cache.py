"""Cache backends: memory + durable Postgres (phase 2).

The Postgres backend replaces Redis on the serverless path, persisting state in
``kv_store`` so it survives across stateless invocations. Tested against an
isolated in-memory SQLite DB (kv_store works on any SQLAlchemy dialect).
"""

from __future__ import annotations

import time

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.cache.memory_cache import MemoryCache
from app.cache.postgres_cache import PostgresCache
from app.db import models  # noqa: F401 - register models (incl. KeyValue)
from app.db.base import Base
from app.db.models import KeyValue


# --------------------------------------------------------------------------- #
# MemoryCache
# --------------------------------------------------------------------------- #
def test_memory_cache_roundtrip_and_delete() -> None:
    c = MemoryCache()
    assert c.get("missing") is None
    c.set("weights", {"historical": 0.5})
    assert c.get("weights") == {"historical": 0.5}
    c.delete("weights")
    assert c.get("weights") is None


def test_memory_cache_expiry() -> None:
    c = MemoryCache()
    c.set("k", 1, ttl=100)
    assert c.get("k") == 1
    # Force the entry to look expired and confirm it's evicted on read.
    c._store["k"] = (1, time.time() - 1)
    assert c.get("k") is None


# --------------------------------------------------------------------------- #
# PostgresCache (durable / serverless)
# --------------------------------------------------------------------------- #
def _pg_cache(monkeypatch):
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    test_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    # PostgresCache imports SessionLocal lazily inside each method.
    monkeypatch.setattr("app.db.base.SessionLocal", test_session)
    return PostgresCache(), test_session


def test_postgres_cache_roundtrip_upsert_and_delete(monkeypatch) -> None:
    cache, _ = _pg_cache(monkeypatch)
    assert cache.get("weights") is None
    cache.set("weights", {"historical": 0.3})
    assert cache.get("weights") == {"historical": 0.3}
    cache.set("weights", {"historical": 0.4})  # upsert, not duplicate
    assert cache.get("weights") == {"historical": 0.4}
    cache.delete("weights")
    assert cache.get("weights") is None


def test_postgres_cache_respects_expiry(monkeypatch) -> None:
    cache, test_session = _pg_cache(monkeypatch)
    with test_session() as s:
        s.add(KeyValue(key="expired", value={"a": 1}, expires_at=time.time() - 10))
        s.add(KeyValue(key="fresh", value={"b": 2}, expires_at=time.time() + 100))
        s.commit()
    assert cache.get("expired") is None  # past expiry -> miss
    assert cache.get("fresh") == {"b": 2}


def test_postgres_cache_persists_via_kv_store(monkeypatch) -> None:
    """The whole point: a fresh cache instance reads what a prior one wrote."""
    cache, _ = _pg_cache(monkeypatch)
    cache.set("orchestration:weights", {"expert": 0.2}, ttl=0)  # ttl=0 -> no expiry
    fresh = PostgresCache()
    assert fresh.get("orchestration:weights") == {"expert": 0.2}
