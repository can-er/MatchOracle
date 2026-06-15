"""Connector repository (Sprints 06 & 08)."""

from __future__ import annotations

from sqlalchemy import select

from app.db.models import Connector
from app.repositories.base import BaseRepository


class ConnectorRepository(BaseRepository[Connector]):
    model = Connector

    def get_by_name(self, name: str, tenant_id: str | None = None) -> Connector | None:
        stmt = select(Connector).where(Connector.name == name)
        if tenant_id is not None:
            stmt = stmt.where(Connector.tenant_id == tenant_id)
        return self.session.scalars(stmt).first()

    def list_all(self, tenant_id: str | None = None) -> list[Connector]:
        stmt = select(Connector).order_by(Connector.created_at.desc())
        if tenant_id is not None:
            stmt = stmt.where(Connector.tenant_id == tenant_id)
        return list(self.session.scalars(stmt).all())
