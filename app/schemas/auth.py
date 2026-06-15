"""Auth / RBAC API schemas (Sprint 13)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.security.principal import ROLE_RANK


class UserCreate(BaseModel):
    username: str
    password: str = Field(min_length=6)
    role: str = Field(default="viewer")
    tenant: str | None = None

    def validated_role(self) -> str:
        role = self.role.strip().lower()
        return role if role in ROLE_RANK else "viewer"


class UserRead(BaseModel):
    username: str
    role: str
    tenant: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    tenant: str


class LoginRequest(BaseModel):
    username: str
    password: str
    tenant: str | None = None


class PrincipalRead(BaseModel):
    username: str
    role: str
    tenant: str
    anonymous: bool


class AuditLogRead(BaseModel):
    actor: str | None = None
    action: str
    resource: str | None = None
    tenant_id: str | None = None
