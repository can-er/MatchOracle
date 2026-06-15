"""ORM models (Sprint 01, extended through later sprints).

Core entities from the data model: ``predictions``, ``agent_results``,
``connectors``. Later sprints add ``outcomes`` (S10), ``accuracy_snapshots``
(S10), ``feedback`` (S12), ``tenants``/``users`` (S13) and ``audit_logs`` (S13).
All carry a nullable ``tenant_id`` so multi-tenancy can be layered on without a
schema break.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Float, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import GUID, Base, JSONType, utcnow


class TenantMixin:
    """Adds a tenant scope column (nullable until Sprint 13 enforces it)."""

    tenant_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)


class Prediction(Base, TenantMixin):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    entity: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    domain: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prediction: Mapped[str] = mapped_column(String(64), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    risk_level: Mapped[str | None] = mapped_column(String(32), nullable=True)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    contributors: Mapped[list | None] = mapped_column(JSONType, nullable=True)
    weights_used: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    # Final-score prediction (Poisson model): scoreline, expected goals, outcome
    # probabilities, top scorelines. Null for non-matchup entities.
    score_detail: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)

    agent_results: Mapped[list[AgentResult]] = relationship(
        back_populates="prediction",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    outcome: Mapped[Outcome | None] = relationship(
        back_populates="prediction",
        cascade="all, delete-orphan",
        uselist=False,
    )


class AgentResult(Base, TenantMixin):
    __tablename__ = "agent_results"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    agent_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    extra: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    prediction: Mapped[Prediction] = relationship(back_populates="agent_results")


class Connector(Base, TenantMixin):
    __tablename__ = "connectors"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    type: Mapped[str] = mapped_column(String(32), nullable=False)  # rest|graphql|sql|nosql|mcp
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="inactive")
    configuration: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)
    last_checked_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    __table_args__ = (UniqueConstraint("name", "tenant_id", name="uq_connector_name_tenant"),)


class Outcome(Base, TenantMixin):
    """Real-world result tied to a prediction (Sprint 10)."""

    __tablename__ = "outcomes"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    actual: Mapped[str] = mapped_column(String(64), nullable=False)
    actual_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    correct: Mapped[bool | None] = mapped_column(nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(default=utcnow)

    prediction: Mapped[Prediction] = relationship(back_populates="outcome")


class AccuracySnapshot(Base, TenantMixin):
    """Time-series accuracy point per agent/global (Sprint 10 / TimescaleDB)."""

    __tablename__ = "accuracy_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    agent_name: Mapped[str] = mapped_column(String(64), nullable=False)  # "__global__" for overall
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    calibration: Mapped[float | None] = mapped_column(Float, nullable=True)
    sample_size: Mapped[int] = mapped_column(default=0)
    weight_after: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)

    __table_args__ = (Index("ix_accuracy_agent_time", "agent_name", "created_at"),)


class Feedback(Base, TenantMixin):
    """Human-in-the-loop validation / correction (Sprint 12)."""

    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    validator: Mapped[str | None] = mapped_column(String(128), nullable=True)
    verdict: Mapped[str] = mapped_column(String(32), nullable=False)  # approve|reject|correct
    corrected_prediction: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class User(Base, TenantMixin):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(128), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(256), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="viewer")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    __table_args__ = (UniqueConstraint("username", "tenant_id", name="uq_user_name_tenant"),)


class AuditLog(Base, TenantMixin):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    actor: Mapped[str | None] = mapped_column(String(128), nullable=True)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    resource: Mapped[str | None] = mapped_column(String(256), nullable=True)
    detail: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)


class KeyValue(Base):
    """Generic durable key-value store (Vercel/Supabase migration, phase 2).

    Backs the Postgres cache backend so process-global state - the agent weight
    vector today, the rotating MPP token later - survives across stateless
    serverless invocations, replacing Redis. ``expires_at`` is epoch seconds
    (DB-agnostic, no timezone pitfalls).
    """

    __tablename__ = "kv_store"

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    value: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    expires_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)
