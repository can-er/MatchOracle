"""Explanation assembly (Sprint 05).

Grounds the human-facing explanation on the quantified agent contributions
(mitigating LLM hallucination per the sprint's risk note). If the Expert agent
produced an LLM explanation it is used as the narrative; otherwise a templated,
fully-grounded sentence is generated.
"""

from __future__ import annotations

from app.orchestration.aggregation import AggregationResult
from app.schemas.agent import AgentResult


def build_explanation(agg: AggregationResult, results: list[AgentResult]) -> str:
    expert = next((r for r in results if r.agent == "expert"), None)
    narrative = ""
    if expert and expert.extra.get("explanation") and not expert.extra.get("llm_fallback"):
        narrative = expert.extra["explanation"].strip()

    top = ", ".join(f"{a} ({agg.per_agent.get(a, 0):.2f})" for a in agg.contributors[:3])
    grounded = (
        f"Predicted '{agg.prediction}' with {agg.confidence_label.lower()} confidence "
        f"({agg.confidence:.2f}). Driven by: {top}."
    )
    if agg.conflict:
        grounded += " Note: agents disagreed, confidence was reduced accordingly."
    if narrative:
        return f"{narrative} ({grounded})"
    return grounded
