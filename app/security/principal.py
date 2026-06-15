"""Request principal + RBAC dependencies (Sprint 13).

When ``auth_enabled`` is off (default) every request runs as a system admin in
the configured default tenant, so open/demo deployments and tests are unchanged.
When it's on, a valid JWT is required and roles are enforced with a simple,
extensible hierarchy: ``viewer`` < ``analyst`` < ``admin``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jwt
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings
from app.security.tokens import decode_access_token

# Role hierarchy — higher rank implies every lower permission.
ROLE_RANK = {"viewer": 0, "analyst": 1, "admin": 2}

_bearer = HTTPBearer(auto_error=False)


@dataclass
class Principal:
    username: str
    role: str
    tenant: str
    anonymous: bool = False

    def has_role(self, minimum: str) -> bool:
        return ROLE_RANK.get(self.role, -1) >= ROLE_RANK.get(minimum, 99)


def _system_principal(tenant: str | None) -> Principal:
    return Principal(
        username="system",
        role="admin",
        tenant=tenant or settings.default_tenant,
        anonymous=True,
    )


def current_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    x_tenant_id: str | None = Header(default=None),
) -> Principal:
    """Resolve the caller. Open-mode → system admin; secured-mode → verified JWT."""
    if not settings.auth_enabled:
        return _system_principal(x_tenant_id)
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        claims = decode_access_token(credentials.credentials)
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    return Principal(
        username=claims.get("sub", "unknown"),
        role=claims.get("role", "viewer"),
        tenant=claims.get("tenant", settings.default_tenant),
    )


def require_role(minimum: str):
    """Dependency factory: require at least ``minimum`` role (no-op when auth off)."""

    def _guard(principal: Principal = Depends(current_principal)) -> Principal:
        if not settings.auth_enabled:
            return principal
        if not principal.has_role(minimum):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Requires '{minimum}' role (you are '{principal.role}')",
            )
        return principal

    return _guard
