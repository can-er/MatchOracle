"""Generic repository base (Sprint 01)."""

from __future__ import annotations

from typing import Generic, TypeVar

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """Thin CRUD wrapper around a SQLAlchemy session for one model."""

    model: type[ModelT]

    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, instance: ModelT) -> ModelT:
        self.session.add(instance)
        self.session.flush()
        return instance

    def get(self, id_) -> ModelT | None:
        return self.session.get(self.model, id_)

    def list(self, *, limit: int = 50, offset: int = 0) -> list[ModelT]:
        stmt = (
            select(self.model)
            .order_by(self.model.created_at.desc())  # type: ignore[attr-defined]
            .limit(limit)
            .offset(offset)
        )
        return list(self.session.scalars(stmt).all())

    def count(self) -> int:
        return int(self.session.scalar(select(func.count()).select_from(self.model)) or 0)

    def delete(self, instance: ModelT) -> None:
        self.session.delete(instance)
        self.session.flush()
