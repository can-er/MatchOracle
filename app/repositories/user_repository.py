"""User & audit-log repositories (Sprint 13)."""

from __future__ import annotations

from sqlalchemy import select

from app.db.models import AuditLog, User
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    model = User

    def by_username(self, username: str, tenant_id: str | None) -> User | None:
        stmt = select(User).where(User.username == username, User.tenant_id == tenant_id)
        return self.session.scalars(stmt).first()

    def count(self) -> int:
        return len(list(self.session.scalars(select(User)).all()))


class AuditLogRepository(BaseRepository[AuditLog]):
    model = AuditLog

    def recent(self, *, limit: int = 100, tenant_id: str | None = None) -> list[AuditLog]:
        stmt = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
        if tenant_id is not None:
            stmt = stmt.where(AuditLog.tenant_id == tenant_id)
        return list(self.session.scalars(stmt).all())
