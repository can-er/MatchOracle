"""Accuracy / benchmarking / weighting API schemas (Sprints 10 & 11)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class OutcomeCreate(BaseModel):
    """Record the real-world result of a prediction (story 10-1)."""

    actual: str = Field(description="Realised outcome, e.g. 'Positive', 'Negative' or '2-1'.")
    actual_score: float | None = Field(default=None, description="Optional numeric realised score.")
    notes: str | None = None


class OutcomeRead(BaseModel):
    prediction_id: str
    actual: str
    actual_score: float | None = None
    correct: bool | None = None
    notes: str | None = None


class AgentBenchmarkRead(BaseModel):
    agent: str
    samples: int
    accuracy: float
    mean_confidence: float
    calibration_error: float
    mean_weight: float
    contribution: float
    flag: str | None = None


class BenchmarkReport(BaseModel):
    evaluated: int = Field(description="Predictions with a usable realised label.")
    agents: list[AgentBenchmarkRead]


class WeightsResponse(BaseModel):
    weights: dict[str, float]


class RecalculateResponse(BaseModel):
    adjusted: bool = Field(description="False when auto-weighting is off or there's no data.")
    samples: int
    accuracy: dict[str, float]
    weights: dict[str, float]


class AutoTuneResponse(BaseModel):
    ingested: int = Field(description="Real World Cup results auto-attached this pass.")
    adjusted: bool
    samples: int
    accuracy: dict[str, float]
