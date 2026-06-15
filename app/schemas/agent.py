"""Agent contract schemas (Sprint 02).

``AgentResult`` is the normalised output every agent must return
(``score`` ∈ [0,1], ``confidence`` ∈ [0,1]) per the Multi-Agent System spec.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AgentResult(BaseModel):
    """Normalised agent output — interchangeable, weightable, benchmarkable."""

    agent: str
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str | None = None
    # Agent-specific extras (e.g. risk_level, recommendation, sources).
    extra: dict = Field(default_factory=dict)


class AgentResultRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    prediction_id: uuid.UUID
    agent_name: str
    score: float
    confidence: float
    weight: float | None = None
    reasoning: str | None = None
    extra: dict | None = None
    created_at: datetime
