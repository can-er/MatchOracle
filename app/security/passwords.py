"""Password hashing (Sprint 13).

passlib CryptContext; defaults to ``pbkdf2_sha256`` (pure-Python, no native
dependency surprises) with ``bcrypt`` accepted for verification. Passwords are
only ever stored hashed — never in plaintext (Sprint 13 DoD).
"""

from __future__ import annotations

from passlib.context import CryptContext

_pwd = CryptContext(schemes=["pbkdf2_sha256", "bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return _pwd.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    try:
        return _pwd.verify(password, hashed)
    except (ValueError, TypeError):
        return False
