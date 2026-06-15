"""JWT access tokens (Sprint 13).

Stateless OAuth2 bearer tokens carrying the subject, role and tenant. Signed with
the secret resolved by the secret backend (env or Vault), never a literal.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt

from app.config import settings
from app.security.secrets import jwt_secret


def create_access_token(
    username: str, role: str, tenant: str, *, expires_minutes: int | None = None
) -> str:
    now = datetime.now(UTC)
    expire = now + timedelta(minutes=expires_minutes or settings.jwt_expire_minutes)
    payload = {
        "sub": username,
        "role": role,
        "tenant": tenant,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    return jwt.encode(payload, jwt_secret(), algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    """Decode/verify a token. Raises ``jwt.PyJWTError`` on any problem."""
    return jwt.decode(token, jwt_secret(), algorithms=[settings.jwt_algorithm])
