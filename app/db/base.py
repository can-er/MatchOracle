"""SQLAlchemy engine, session factory and declarative base (Sprint 01).

A cross-dialect ``GUID`` type lets the same models run on PostgreSQL (native
``UUID``) and SQLite (``CHAR(36)``) so tests and CI need no database server.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

from sqlalchemy import CHAR, DateTime, TypeDecorator, create_engine
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


def utcnow() -> datetime:
    """Timezone-aware UTC now (avoids deprecated ``datetime.utcnow``)."""
    return datetime.now(UTC)


class GUID(TypeDecorator):
    """Platform-independent UUID: PostgreSQL ``UUID`` else ``CHAR(36)``."""

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


class JSONType(TypeDecorator):
    """JSONB on PostgreSQL, generic JSON elsewhere."""

    impl = JSONB
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        from sqlalchemy import JSON

        return dialect.type_descriptor(JSON())


class TZDateTime(DateTime):
    """DateTime that is always timezone-aware."""


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def _make_engine():
    connect_args = {"check_same_thread": False} if settings.is_sqlite else {}
    return create_engine(
        settings.database_url,
        echo=False,
        future=True,
        pool_pre_ping=not settings.is_sqlite,
        connect_args=connect_args,
    )


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a scoped session."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all() -> None:
    """Create tables directly (used for SQLite/tests; prod uses Alembic)."""
    # Import models so they register on the metadata before create_all.
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
