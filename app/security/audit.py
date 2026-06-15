"""Audit logging of sensitive actions (Sprint 13, story 13-4)."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import AuditLog
from app.security.principal import Principal


def tenant_scope(principal: Principal) -> str | None:
    """The tenant a request is bound to — ``None`` in open mode (legacy behaviour)."""
    return principal.tenant if settings.auth_enabled else None


def record_audit(
    session: Session,
    actor: str | None,
    action: str,
    *,
    resource: str | None = None,
    detail: dict | None = None,
    tenant_id: str | None = None,
) -> None:
    """Append an audit row. Best-effort: never raises into the request path."""
    try:
        session.add(
            AuditLog(
                actor=actor,
                action=action,
                resource=resource,
                detail=detail,
                tenant_id=tenant_id,
            )
        )
        session.flush()
    except Exception:  # auditing must not break the operation it records
        session.rollback()
