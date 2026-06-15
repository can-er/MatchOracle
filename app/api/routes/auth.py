"""Authentication, user management & audit endpoints (Sprint 13).

OAuth2-ish bearer flow: register users, exchange credentials for a JWT, inspect
the current principal, and read the audit trail. The very first user can be
bootstrapped without auth (and is forced to ``admin``); afterwards, creating
users requires an admin principal.
"""

from __future__ import annotations

import jwt
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config import settings
from app.db.base import get_session
from app.db.models import User
from app.repositories.user_repository import AuditLogRepository, UserRepository
from app.schemas.auth import (
    AuditLogRead,
    LoginRequest,
    PrincipalRead,
    TokenResponse,
    UserCreate,
    UserRead,
)
from app.security.audit import record_audit, tenant_scope
from app.security.passwords import hash_password, verify_password
from app.security.principal import Principal, current_principal, require_role
from app.security.tokens import create_access_token, decode_access_token

router = APIRouter(tags=["auth"])

_bearer = HTTPBearer(auto_error=False)


def _is_admin(creds: HTTPAuthorizationCredentials | None) -> bool:
    if creds is None or not creds.credentials:
        return False
    try:
        return decode_access_token(creds.credentials).get("role") == "admin"
    except jwt.PyJWTError:
        return False


@router.post(
    "/auth/register",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a user (first user bootstraps as admin; then admin-only)",
)
def register(
    payload: UserCreate,
    session: Session = Depends(get_session),
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> UserRead:
    repo = UserRepository(session)
    bootstrap = repo.count() == 0
    if not bootstrap and settings.auth_enabled and not _is_admin(creds):
        # Only an admin may create further users once auth is enforced.
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin role required to create users")
    tenant = payload.tenant or settings.default_tenant
    role = "admin" if bootstrap else payload.validated_role()
    if repo.by_username(payload.username, tenant) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "User already exists in this tenant")
    user = User(
        username=payload.username,
        hashed_password=hash_password(payload.password),
        role=role,
        tenant_id=tenant,
    )
    repo.add(user)
    record_audit(session, payload.username, "user.register", resource=role, tenant_id=tenant)
    return UserRead(username=user.username, role=user.role, tenant=user.tenant_id)


@router.post("/auth/token", response_model=TokenResponse, summary="Exchange credentials for a JWT")
def login(payload: LoginRequest, session: Session = Depends(get_session)) -> TokenResponse:
    tenant = payload.tenant or settings.default_tenant
    user = UserRepository(session).by_username(payload.username, tenant)
    if user is None or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    token = create_access_token(user.username, user.role, tenant)
    record_audit(session, user.username, "auth.login", tenant_id=tenant)
    return TokenResponse(access_token=token, role=user.role, tenant=tenant)


@router.get("/auth/me", response_model=PrincipalRead, summary="Inspect the current principal")
def me(principal: Principal = Depends(current_principal)) -> PrincipalRead:
    return PrincipalRead(
        username=principal.username,
        role=principal.role,
        tenant=principal.tenant,
        anonymous=principal.anonymous,
    )


@router.get(
    "/audit",
    response_model=list[AuditLogRead],
    summary="Read the audit trail (admin)",
)
def list_audit(
    limit: int = Query(100, ge=1, le=500),
    session: Session = Depends(get_session),
    principal: Principal = Depends(require_role("admin")),
) -> list[AuditLogRead]:
    rows = AuditLogRepository(session).recent(limit=limit, tenant_id=tenant_scope(principal))
    return [
        AuditLogRead(actor=r.actor, action=r.action, resource=r.resource, tenant_id=r.tenant_id)
        for r in rows
    ]
