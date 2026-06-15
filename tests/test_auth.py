"""Auth, RBAC, multi-tenant isolation & audit (Sprint 13).

Builds an isolated app instance with its own in-memory DB so toggling
``auth_enabled`` here never leaks into the other API tests.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.db import models  # noqa: F401
from app.db.base import Base, get_session
from app.main import create_app
from app.security.passwords import hash_password, verify_password
from app.security.tokens import create_access_token, decode_access_token

_engine = create_engine(
    "sqlite+pysqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_Session = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
Base.metadata.create_all(_engine)


def _override() -> Iterator[Session]:
    session = _Session()
    try:
        yield session
        session.commit()
    finally:
        session.close()


app = create_app()
app.dependency_overrides[get_session] = _override
client = TestClient(app)


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def test_password_hash_roundtrip() -> None:
    h = hash_password("s3cret!")
    assert h != "s3cret!"  # never stored in plaintext
    assert verify_password("s3cret!", h)
    assert not verify_password("wrong", h)


def test_token_roundtrip() -> None:
    token = create_access_token("alice", "admin", "acme")
    claims = decode_access_token(token)
    assert claims["sub"] == "alice"
    assert claims["role"] == "admin"
    assert claims["tenant"] == "acme"


# --------------------------------------------------------------------------- #
# Bootstrap + login + me (open mode)
# --------------------------------------------------------------------------- #
def test_bootstrap_admin_then_login_and_me() -> None:
    # First user bootstraps as admin regardless of requested role.
    reg = client.post(
        "/api/v1/auth/register",
        json={"username": "root", "password": "secret123", "role": "viewer", "tenant": "ta"},
    )
    assert reg.status_code == 201, reg.text
    assert reg.json()["role"] == "admin"

    tok = client.post(
        "/api/v1/auth/token",
        json={"username": "root", "password": "secret123", "tenant": "ta"},
    )
    assert tok.status_code == 200
    token = tok.json()["access_token"]

    # In open mode (auth off), /me reports the permissive system principal.
    me = client.get("/api/v1/auth/me", headers=_auth(token))
    assert me.status_code == 200
    assert me.json()["username"] == "system"
    assert me.json()["anonymous"] is True


def test_login_wrong_password_401() -> None:
    bad = client.post(
        "/api/v1/auth/token",
        json={"username": "root", "password": "nope", "tenant": "ta"},
    )
    assert bad.status_code == 401


# --------------------------------------------------------------------------- #
# RBAC (auth enforced)
# --------------------------------------------------------------------------- #
def test_rbac_enforced_when_auth_on(monkeypatch: pytest.MonkeyPatch) -> None:
    # Create an analyst and a viewer while auth is still open.
    client.post(
        "/api/v1/auth/register",
        json={"username": "ana", "password": "secret123", "role": "analyst", "tenant": "ta"},
    )
    client.post(
        "/api/v1/auth/register",
        json={"username": "vic", "password": "secret123", "role": "viewer", "tenant": "ta"},
    )
    analyst = client.post(
        "/api/v1/auth/token", json={"username": "ana", "password": "secret123", "tenant": "ta"}
    ).json()["access_token"]
    viewer = client.post(
        "/api/v1/auth/token", json={"username": "vic", "password": "secret123", "tenant": "ta"}
    ).json()["access_token"]

    monkeypatch.setattr(settings, "auth_enabled", True)

    # No token -> 401.
    assert client.post("/api/v1/predict", json={"entity": "X"}).status_code == 401
    # Viewer can't predict -> 403.
    r = client.post("/api/v1/predict", json={"entity": "X"}, headers=_auth(viewer))
    assert r.status_code == 403
    # Analyst can.
    ok = client.post("/api/v1/predict", json={"entity": "X"}, headers=_auth(analyst))
    assert ok.status_code == 201, ok.text


# --------------------------------------------------------------------------- #
# Multi-tenant isolation (Sprint 13 DoD)
# --------------------------------------------------------------------------- #
def test_tenant_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two admins in two tenants (created while auth is open).
    client.post(
        "/api/v1/auth/register",
        json={"username": "ba", "password": "secret123", "role": "admin", "tenant": "alpha"},
    )
    client.post(
        "/api/v1/auth/register",
        json={"username": "bb", "password": "secret123", "role": "admin", "tenant": "beta"},
    )
    ta = client.post(
        "/api/v1/auth/token", json={"username": "ba", "password": "secret123", "tenant": "alpha"}
    ).json()["access_token"]
    tb = client.post(
        "/api/v1/auth/token", json={"username": "bb", "password": "secret123", "tenant": "beta"}
    ).json()["access_token"]

    monkeypatch.setattr(settings, "auth_enabled", True)

    a_pred = client.post(
        "/api/v1/predict", json={"entity": "Alpha Co"}, headers=_auth(ta)
    ).json()["id"]
    client.post("/api/v1/predict", json={"entity": "Beta Co"}, headers=_auth(tb))

    # Each tenant lists only its own predictions.
    a_list = client.get("/api/v1/predictions", headers=_auth(ta)).json()
    assert all(item["entity"] != "Beta Co" for item in a_list["items"])
    b_list = client.get("/api/v1/predictions", headers=_auth(tb)).json()
    assert all(item["entity"] != "Alpha Co" for item in b_list["items"])

    # Tenant beta cannot fetch tenant alpha's prediction by id -> 404.
    cross = client.get(f"/api/v1/predictions/{a_pred}", headers=_auth(tb))
    assert cross.status_code == 404
    # But alpha can.
    own = client.get(f"/api/v1/predictions/{a_pred}", headers=_auth(ta))
    assert own.status_code == 200


# --------------------------------------------------------------------------- #
# Audit trail (story 13-4)
# --------------------------------------------------------------------------- #
def test_audit_trail_records_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    admin = client.post(
        "/api/v1/auth/token", json={"username": "root", "password": "secret123", "tenant": "ta"}
    ).json()["access_token"]

    monkeypatch.setattr(settings, "auth_enabled", True)
    client.post("/api/v1/predict", json={"entity": "Audited Co"}, headers=_auth(admin))

    audit = client.get("/api/v1/audit", headers=_auth(admin))
    assert audit.status_code == 200
    actions = {row["action"] for row in audit.json()}
    assert "prediction.create" in actions
