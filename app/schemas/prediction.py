"""Prediction request/response schemas (Sprints 05 & 06)."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.agent import AgentResultRead


class PredictionRequest(BaseModel):
    entity: str = Field(min_length=1, max_length=512, examples=["Target Entity"])
    domain: str | None = Field(default=None, examples=["sports", "finance", "worldcup"])
    context: dict = Field(default_factory=dict, description="Optional extra context for agents")
    knockout: bool = Field(
        default=False,
        description="Knockout match: force a winner (extra time / penalties) on a draw.",
    )


class PredictionResponse(BaseModel):
    """Enriched, explainable response (matches the API REST spec)."""

    id: uuid.UUID
    entity: str
    prediction: str
    score: float
    confidence: float
    risk_level: str | None = None
    contributors: list[str] = Field(default_factory=list)
    explanation: str | None = None
    # Final-score prediction (matchups only): scoreline, expected goals, outcome
    # probabilities, top scorelines. Null for non-matchup entities.
    score_detail: dict | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @field_validator("contributors", mode="before")
    @classmethod
    def _none_to_list(cls, value: object) -> object:
        # Score-only snapshots (e.g. the scheduled refresh) have no contributors.
        return value or []


class PredictionDetail(PredictionResponse):
    weights_used: dict | None = None
    agent_results: list[AgentResultRead] = Field(default_factory=list)


class OutcomeCreate(BaseModel):
    """Record the real-world result of a prediction (Sprint 10)."""

    actual: str = Field(examples=["Positive", "Negative"])
    actual_score: float | None = Field(default=None, ge=0.0, le=1.0)
    notes: str | None = None


class OutcomeRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    prediction_id: uuid.UUID
    actual: str
    actual_score: float | None = None
    correct: bool | None = None
    recorded_at: datetime
