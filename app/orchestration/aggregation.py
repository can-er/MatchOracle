"""Aggregation, confidence and conflict resolution (Sprint 05).

Turns the agents' normalised outputs into a final, explainable prediction:
``Weighted Prediction = Σ (wᵢ · scoreᵢ)`` and a global confidence that blends
weighted agent confidence with inter-agent agreement.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from app.schemas.agent import AgentResult

# Confidence banding for the human-facing label.
CONFIDENCE_BANDS = [(0.75, "High"), (0.5, "Medium"), (0.0, "Low")]


@dataclass
class AggregationResult:
    prediction: str
    score: float
    confidence: float
    confidence_label: str
    risk_level: str | None
    contributors: list[str]
    weights_used: dict[str, float]
    conflict: bool = False
    per_agent: dict[str, float] = field(default_factory=dict)


def _confidence_label(value: float) -> str:
    for threshold, label in CONFIDENCE_BANDS:
        if value >= threshold:
            return label
    return "Low"


def _agreement(scores: list[float]) -> float:
    """1.0 = perfect agreement, →0 as dispersion grows."""
    if len(scores) < 2:
        return 1.0
    spread = statistics.pstdev(scores)
    # stdev of 0.5 (max for [0,1] bimodal) maps to ~0 agreement.
    return max(0.0, 1.0 - spread * 2.0)


def aggregate(
    results: list[AgentResult],
    weights: dict[str, float],
    *,
    conflict_threshold: float = 0.35,
) -> AggregationResult:
    """Combine agent results into a final prediction.

    Conflict resolution: when agents strongly disagree (score spread beyond
    ``conflict_threshold``) confidence is penalised and the result is flagged so
    callers/the explanation can surface the disagreement.
    """
    if not results:
        raise ValueError("Cannot aggregate an empty result set")

    # Restrict weights to present agents and renormalise.
    present = {r.agent for r in results}
    active_weights = {a: weights.get(a, 0.0) for a in present}
    if sum(active_weights.values()) <= 0:
        active_weights = {a: 1.0 / len(present) for a in present}
    total_w = sum(active_weights.values())
    active_weights = {a: w / total_w for a, w in active_weights.items()}

    weighted_score = sum(active_weights[r.agent] * r.score for r in results)
    weighted_conf = sum(active_weights[r.agent] * r.confidence for r in results)

    scores = [r.score for r in results]
    agreement = _agreement(scores)
    spread = (max(scores) - min(scores)) if len(scores) > 1 else 0.0
    conflict = spread >= conflict_threshold

    confidence = weighted_conf * (0.6 + 0.4 * agreement)
    if conflict:
        confidence *= 0.85
    confidence = round(min(1.0, max(0.0, confidence)), 3)

    # Contributors = agents pushing in the winning direction, by weighted impact.
    direction_positive = weighted_score >= 0.5
    contributors = [
        r.agent
        for r in sorted(results, key=lambda r: active_weights[r.agent] * r.score, reverse=True)
        if (r.score >= 0.5) == direction_positive and r.agent != "expert"
    ]

    risk_level = None
    for r in results:
        if r.agent == "risk":
            risk_level = r.extra.get("risk_level")

    prediction_label = "Positive" if direction_positive else "Negative"

    return AggregationResult(
        prediction=prediction_label,
        score=round(weighted_score, 3),
        confidence=confidence,
        confidence_label=_confidence_label(confidence),
        risk_level=risk_level,
        contributors=contributors or [r.agent for r in results if r.agent != "expert"],
        weights_used={a: round(w, 4) for a, w in active_weights.items()},
        conflict=conflict,
        per_agent={r.agent: r.score for r in results},
    )
