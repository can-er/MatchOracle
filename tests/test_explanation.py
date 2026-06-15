"""Sprint 05 explainability DoD tests for build_explanation.

These assert the *contract/behaviour* of the grounded explanation assembly:
- the prediction label and confidence value+label are present,
- the top contributors are mentioned,
- the Expert narrative is woven in only when it is a real (non-fallback)
  explanation, and
- the disagreement note is appended when the aggregation flagged a conflict.
"""

from __future__ import annotations

import pytest

from app.orchestration.aggregation import AggregationResult
from app.orchestration.explanation import build_explanation
from app.schemas.agent import AgentResult


def _agg(
    *,
    prediction: str = "Positive",
    score: float = 0.71,
    confidence: float = 0.82,
    confidence_label: str = "High",
    contributors: list[str] | None = None,
    per_agent: dict[str, float] | None = None,
    conflict: bool = False,
    risk_level: str | None = None,
) -> AggregationResult:
    """Build an AggregationResult with sensible, internally-consistent defaults."""
    contributors = contributors if contributors is not None else ["historical", "trend", "market"]
    per_agent = (
        per_agent if per_agent is not None else {"historical": 0.74, "trend": 0.67, "market": 0.72}
    )
    return AggregationResult(
        prediction=prediction,
        score=score,
        confidence=confidence,
        confidence_label=confidence_label,
        risk_level=risk_level,
        contributors=contributors,
        weights_used={a: round(1.0 / len(per_agent), 4) for a in per_agent},
        conflict=conflict,
        per_agent=per_agent,
    )


def _result(
    agent: str,
    score: float = 0.7,
    confidence: float = 0.7,
    *,
    extra: dict | None = None,
) -> AgentResult:
    return AgentResult(
        agent=agent,
        score=score,
        confidence=confidence,
        reasoning=None,
        extra=extra or {},
    )


# --------------------------------------------------------------------------- #
# Grounded text: prediction label, confidence value + label, contributors.
# --------------------------------------------------------------------------- #


def test_explanation_contains_prediction_label():
    agg = _agg(prediction="Positive")
    text = build_explanation(agg, [])
    assert "Positive" in text


def test_explanation_contains_negative_prediction_label():
    agg = _agg(prediction="Negative")
    text = build_explanation(agg, [])
    assert "Negative" in text


def test_explanation_contains_confidence_value_formatted():
    agg = _agg(confidence=0.82)
    text = build_explanation(agg, [])
    # The grounded sentence formats confidence with two decimals.
    assert "0.82" in text


def test_explanation_contains_confidence_label_lowercased():
    agg = _agg(confidence_label="High")
    text = build_explanation(agg, [])
    # build_explanation lowercases the band label in the grounded sentence.
    assert "high confidence" in text


def test_explanation_mentions_top_contributors():
    contributors = ["historical", "trend", "market"]
    per_agent = {"historical": 0.74, "trend": 0.67, "market": 0.72}
    agg = _agg(contributors=contributors, per_agent=per_agent)
    text = build_explanation(agg, [])
    for name in contributors:
        assert name in text


def test_explanation_includes_contributor_scores():
    per_agent = {"historical": 0.74, "trend": 0.67, "market": 0.72}
    agg = _agg(contributors=list(per_agent), per_agent=per_agent)
    text = build_explanation(agg, [])
    # Each contributor's per-agent score is rendered with two decimals.
    for value in per_agent.values():
        assert f"{value:.2f}" in text


def test_explanation_limits_to_top_three_contributors():
    contributors = ["historical", "trend", "market", "contextual", "risk"]
    per_agent = {
        "historical": 0.80,
        "trend": 0.70,
        "market": 0.60,
        "contextual": 0.50,
        "risk": 0.40,
    }
    agg = _agg(contributors=contributors, per_agent=per_agent)
    text = build_explanation(agg, [])
    # Only the first three contributors should appear in "Driven by:".
    for name in contributors[:3]:
        assert name in text
    for name in contributors[3:]:
        assert name not in text


def test_explanation_uses_zero_default_for_missing_per_agent_score():
    # Contributor present but absent from per_agent -> formatted as 0.00.
    agg = _agg(contributors=["ghost"], per_agent={})
    text = build_explanation(agg, [])
    assert "ghost (0.00)" in text


def test_explanation_returns_non_empty_string():
    text = build_explanation(_agg(), [])
    assert isinstance(text, str)
    assert text.strip()


# --------------------------------------------------------------------------- #
# Expert narrative inclusion rules.
# --------------------------------------------------------------------------- #


def test_expert_explanation_included_when_present_and_not_fallback():
    narrative = "Recent trends and historical patterns point toward a favorable outcome."
    expert = _result("expert", extra={"explanation": narrative})
    text = build_explanation(_agg(), [expert])
    assert narrative in text
    # Grounded sentence is still appended (in parentheses) after the narrative.
    assert "high confidence" in text
    assert text.startswith(narrative)


def test_expert_explanation_excluded_when_llm_fallback_flag_set():
    narrative = "This is a generic fallback narrative."
    expert = _result("expert", extra={"explanation": narrative, "llm_fallback": True})
    text = build_explanation(_agg(), [expert])
    assert narrative not in text
    # Falls back to the purely grounded sentence.
    assert text.startswith("Predicted")


def test_expert_explanation_excluded_when_explanation_empty():
    expert = _result("expert", extra={"explanation": ""})
    text = build_explanation(_agg(), [expert])
    assert text.startswith("Predicted")


def test_expert_explanation_excluded_when_no_explanation_key():
    expert = _result("expert", extra={"recommendation": "positive"})
    text = build_explanation(_agg(), [expert])
    assert text.startswith("Predicted")


def test_non_expert_explanation_ignored():
    # An "explanation" extra on a non-expert agent must not be used.
    other = _result("historical", extra={"explanation": "should not appear"})
    text = build_explanation(_agg(), [other])
    assert "should not appear" not in text


def test_expert_narrative_is_stripped():
    narrative = "   Surrounded by whitespace.   "
    expert = _result("expert", extra={"explanation": narrative})
    text = build_explanation(_agg(), [expert])
    assert text.startswith("Surrounded by whitespace.")
    assert "   Surrounded" not in text


def test_grounded_sentence_wrapped_in_parens_when_narrative_present():
    narrative = "Narrative text."
    expert = _result("expert", extra={"explanation": narrative})
    text = build_explanation(_agg(), [expert])
    # Format: "<narrative> (<grounded>)"
    assert "(Predicted 'Positive'" in text
    assert text.rstrip().endswith(")")


# --------------------------------------------------------------------------- #
# Conflict / disagreement note.
# --------------------------------------------------------------------------- #


def test_conflict_appends_disagreement_note():
    agg = _agg(conflict=True)
    text = build_explanation(agg, [])
    assert "agents disagreed" in text
    assert "confidence was reduced" in text


def test_no_conflict_omits_disagreement_note():
    agg = _agg(conflict=False)
    text = build_explanation(agg, [])
    assert "agents disagreed" not in text


def test_conflict_note_present_even_with_narrative():
    narrative = "A confident narrative."
    expert = _result("expert", extra={"explanation": narrative})
    agg = _agg(conflict=True)
    text = build_explanation(agg, [expert])
    assert narrative in text
    assert "agents disagreed" in text


# --------------------------------------------------------------------------- #
# Contract: AgentResult range validation (out-of-range must raise).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_score", [-0.01, 1.01])
def test_agent_result_rejects_out_of_range_score(bad_score):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AgentResult(agent="expert", score=bad_score, confidence=0.5)


@pytest.mark.parametrize("bad_conf", [-0.5, 1.5])
def test_agent_result_rejects_out_of_range_confidence(bad_conf):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AgentResult(agent="expert", score=0.5, confidence=bad_conf)
