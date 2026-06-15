"""Feedback / self-improvement / multi-model API schemas (Sprint 12)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.prediction.feedback import VERDICTS


class FeedbackCreate(BaseModel):
    verdict: str = Field(description="One of: approve | reject | correct.")
    validator: str | None = Field(default=None, description="Who reviewed the prediction.")
    corrected_prediction: str | None = Field(
        default=None, description="The corrected label (for verdict='correct')."
    )
    comment: str | None = None

    @field_validator("verdict")
    @classmethod
    def _known_verdict(cls, value: str) -> str:
        v = value.strip().lower()
        if v not in VERDICTS:
            raise ValueError(f"verdict must be one of {sorted(VERDICTS)}")
        return v


class FeedbackRead(BaseModel):
    id: str
    prediction_id: str
    verdict: str
    validator: str | None = None
    corrected_prediction: str | None = None
    reward: float | None = None
    comment: str | None = None


class LearnResponse(BaseModel):
    adjusted: bool = Field(description="False when there's no usable feedback yet.")
    samples: int
    accuracy: dict[str, float]
    weights: dict[str, float]


class ModelRouteResponse(BaseModel):
    policy: str
    complexity: float
    selected_tier: str
    selected_model: str
    tiers: list[dict]
