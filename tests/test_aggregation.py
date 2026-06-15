"""Sprint 05 aggregation / conflict resolution DoD tests.

These exercise :func:`app.orchestration.aggregation.aggregate` with hand-built
``AgentResult`` lists whose scores are chosen so the weighted maths is exactly
computable by hand. We assert contracts and behaviour (ranges, bands,
determinism, conflict flagging, contributor direction) rather than opaque
internals.
"""

from __future__ import annotations

import statistics

import pytest

from app.orchestration.aggregation import (
    CONFIDENCE_BANDS,
    AggregationResult,
    aggregate,
)
from app.schemas.agent import AgentResult


def _result(agent: str, score: float, confidence: float = 1.0, **extra) -> AgentResult:
    return AgentResult(
        agent=agent,
        score=score,
        confidence=confidence,
        extra=extra or {},
    )


# ---------------------------------------------------------------------------
# Weighted score
# ---------------------------------------------------------------------------


def test_weighted_score_hand_computed():
    """Active weights are restricted-to-present, renormalised, then applied.

    historical=0.8 (w 0.6) and trend=0.4 (w 0.2): only these two present so
    renormalised weights are 0.75 / 0.25. Score = 0.75*0.8 + 0.25*0.4 = 0.7.
    """
    results = [
        _result("historical", 0.8),
        _result("trend", 0.4),
    ]
    weights = {"historical": 0.6, "trend": 0.2}

    out = aggregate(results, weights)

    assert isinstance(out, AggregationResult)
    assert out.score == 0.7
    # Renormalised active weights, rounded to 4dp by the implementation.
    assert out.weights_used == {"historical": 0.75, "trend": 0.25}


def test_weighted_score_three_agents_hand_computed():
    """Three present agents with weights summing < 1 still renormalise to 1."""
    results = [
        _result("historical", 1.0),
        _result("trend", 0.0),
        _result("market", 0.5),
    ]
    # Equal raw weights -> equal renormalised weights 1/3 each.
    weights = {"historical": 0.1, "trend": 0.1, "market": 0.1}

    out = aggregate(results, weights)

    expected = round((1.0 + 0.0 + 0.5) / 3.0, 3)
    assert out.score == expected


def test_score_is_rounded_to_three_dp():
    results = [
        _result("historical", 0.123456),
        _result("trend", 0.654321),
    ]
    weights = {"historical": 0.5, "trend": 0.5}
    out = aggregate(results, weights)
    # Symmetric weights -> mean; rounded to 3dp.
    assert out.score == round((0.123456 + 0.654321) / 2.0, 3)


# ---------------------------------------------------------------------------
# Confidence banding
# ---------------------------------------------------------------------------


def test_confidence_bands_constant_documented():
    """Guard the documented thresholds so the banding test stays meaningful."""
    assert CONFIDENCE_BANDS == [(0.75, "High"), (0.5, "Medium"), (0.0, "Low")]


def test_confidence_label_high():
    # Two agreeing agents with full confidence -> High band.
    results = [
        _result("historical", 0.9, confidence=1.0),
        _result("trend", 0.9, confidence=1.0),
    ]
    weights = {"historical": 1.0, "trend": 1.0}
    out = aggregate(results, weights)
    assert out.confidence >= 0.75
    assert out.confidence_label == "High"


def test_confidence_label_medium():
    # Moderate confidence lands in the [0.5, 0.75) Medium band.
    results = [
        _result("historical", 0.6, confidence=0.7),
        _result("trend", 0.6, confidence=0.7),
    ]
    weights = {"historical": 1.0, "trend": 1.0}
    out = aggregate(results, weights)
    assert 0.5 <= out.confidence < 0.75
    assert out.confidence_label == "Medium"


def test_confidence_label_low():
    # Zero confidence -> Low band.
    results = [
        _result("historical", 0.5, confidence=0.0),
        _result("trend", 0.5, confidence=0.0),
    ]
    weights = {"historical": 1.0, "trend": 1.0}
    out = aggregate(results, weights)
    assert out.confidence < 0.5
    assert out.confidence_label == "Low"


def test_confidence_label_matches_banding_logic_across_results():
    """Whatever confidence comes out, its label must obey CONFIDENCE_BANDS."""
    samples = [
        [_result("historical", 0.95, 1.0), _result("trend", 0.95, 1.0)],
        [_result("historical", 0.6, 0.6), _result("trend", 0.6, 0.6)],
        [_result("historical", 0.5, 0.1), _result("trend", 0.5, 0.1)],
    ]
    for results in samples:
        out = aggregate(results, {"historical": 1.0, "trend": 1.0})
        expected = next(
            label for threshold, label in CONFIDENCE_BANDS if out.confidence >= threshold
        )
        assert out.confidence_label == expected


# ---------------------------------------------------------------------------
# Conflict + agreement
# ---------------------------------------------------------------------------


def test_conflict_flagged_on_high_spread():
    results = [
        _result("historical", 0.05, confidence=1.0),
        _result("trend", 0.95, confidence=1.0),
    ]
    weights = {"historical": 1.0, "trend": 1.0}
    out = aggregate(results, weights)
    # Spread 0.90 >= default conflict_threshold 0.35.
    assert out.conflict is True


def test_agreeing_set_not_in_conflict():
    results = [
        _result("historical", 0.70, confidence=1.0),
        _result("trend", 0.72, confidence=1.0),
    ]
    weights = {"historical": 1.0, "trend": 1.0}
    out = aggregate(results, weights)
    # Spread 0.02 < 0.35.
    assert out.conflict is False


def test_conflict_reduces_confidence_vs_agreement():
    """A divergent set must yield lower confidence than an agreeing set with
    identical per-agent confidence."""
    weights = {"historical": 1.0, "trend": 1.0}

    agreeing = aggregate(
        [
            _result("historical", 0.70, confidence=1.0),
            _result("trend", 0.72, confidence=1.0),
        ],
        weights,
    )
    diverging = aggregate(
        [
            _result("historical", 0.05, confidence=1.0),
            _result("trend", 0.95, confidence=1.0),
        ],
        weights,
    )

    assert agreeing.conflict is False
    assert diverging.conflict is True
    assert diverging.confidence < agreeing.confidence


def test_agreement_near_one_when_scores_close():
    """Close scores -> agreement bonus keeps confidence near the weighted
    confidence (i.e. agreement factor ~1.0)."""
    results = [
        _result("historical", 0.80, confidence=1.0),
        _result("trend", 0.80, confidence=1.0),
    ]
    out = aggregate(results, {"historical": 1.0, "trend": 1.0})
    # Identical scores -> pstdev 0 -> agreement 1.0 -> confidence = weighted_conf
    # = 1.0 (no conflict penalty).
    assert out.conflict is False
    assert out.confidence == 1.0


def test_conflict_threshold_is_configurable():
    results = [
        _result("historical", 0.40, confidence=1.0),
        _result("trend", 0.60, confidence=1.0),
    ]
    weights = {"historical": 1.0, "trend": 1.0}
    # Spread 0.20: below default 0.35 (no conflict) but >= a tight threshold.
    assert aggregate(results, weights).conflict is False
    assert aggregate(results, weights, conflict_threshold=0.1).conflict is True


def test_single_result_never_conflicts():
    out = aggregate([_result("historical", 0.9, confidence=0.8)], {"historical": 1.0})
    assert out.conflict is False


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_empty_results_raise_value_error():
    with pytest.raises(ValueError):
        aggregate([], {"historical": 1.0})


# ---------------------------------------------------------------------------
# Contributors
# ---------------------------------------------------------------------------


def test_contributors_exclude_expert():
    results = [
        _result("historical", 0.9, confidence=1.0),
        _result("trend", 0.8, confidence=1.0),
        _result("expert", 0.95, confidence=1.0),
    ]
    weights = {"historical": 1.0, "trend": 1.0, "expert": 1.0}
    out = aggregate(results, weights)
    assert "expert" not in out.contributors
    assert "historical" in out.contributors
    assert "trend" in out.contributors


def test_contributors_point_in_winning_positive_direction():
    """Winning direction is positive -> only agents with score >= 0.5 contribute."""
    results = [
        _result("historical", 0.9, confidence=1.0),
        _result("trend", 0.85, confidence=1.0),
        _result("market", 0.10, confidence=1.0),  # against the positive winner
    ]
    weights = {"historical": 1.0, "trend": 1.0, "market": 1.0}
    out = aggregate(results, weights)
    assert out.prediction == "Positive"
    assert "historical" in out.contributors
    assert "trend" in out.contributors
    assert "market" not in out.contributors


def test_contributors_point_in_winning_negative_direction():
    results = [
        _result("historical", 0.10, confidence=1.0),
        _result("trend", 0.05, confidence=1.0),
        _result("market", 0.90, confidence=1.0),  # against the negative winner
    ]
    weights = {"historical": 1.0, "trend": 1.0, "market": 1.0}
    out = aggregate(results, weights)
    assert out.prediction == "Negative"
    assert "historical" in out.contributors
    assert "trend" in out.contributors
    assert "market" not in out.contributors


def test_contributors_sorted_by_weighted_impact():
    """Higher weighted impact (weight*score) comes first among contributors."""
    results = [
        _result("trend", 0.6, confidence=1.0),
        _result("historical", 0.9, confidence=1.0),
    ]
    # historical heavier weight AND higher score -> ranks before trend.
    weights = {"historical": 0.8, "trend": 0.2}
    out = aggregate(results, weights)
    assert out.contributors.index("historical") < out.contributors.index("trend")


# ---------------------------------------------------------------------------
# Risk level passthrough
# ---------------------------------------------------------------------------


def test_risk_level_pulled_from_risk_agent_extra():
    results = [
        _result("historical", 0.7, confidence=1.0),
        _result("risk", 0.4, confidence=1.0, risk_level="medium"),
    ]
    weights = {"historical": 1.0, "risk": 1.0}
    out = aggregate(results, weights)
    assert out.risk_level == "medium"


def test_risk_level_none_without_risk_agent():
    results = [
        _result("historical", 0.7, confidence=1.0),
        _result("trend", 0.6, confidence=1.0),
    ]
    out = aggregate(results, {"historical": 1.0, "trend": 1.0})
    assert out.risk_level is None


def test_risk_level_none_when_risk_agent_lacks_extra_key():
    results = [
        _result("historical", 0.7, confidence=1.0),
        _result("risk", 0.4, confidence=1.0),  # no risk_level in extra
    ]
    out = aggregate(results, {"historical": 1.0, "risk": 1.0})
    assert out.risk_level is None


# ---------------------------------------------------------------------------
# Weight renormalisation with a subset present
# ---------------------------------------------------------------------------


def test_active_weights_sum_to_one_with_subset_present():
    """Only a subset of the weighted agents is present -> active weights renorm."""
    results = [
        _result("historical", 0.6, confidence=1.0),
        _result("trend", 0.4, confidence=1.0),
    ]
    # Full weight vector references agents that are NOT present.
    weights = {
        "historical": 0.25,
        "trend": 0.20,
        "contextual": 0.15,
        "risk": 0.15,
        "market": 0.10,
        "expert": 0.15,
    }
    out = aggregate(results, weights)

    assert set(out.weights_used) == {"historical", "trend"}
    assert sum(out.weights_used.values()) == pytest.approx(1.0, abs=1e-3)


def test_zero_total_weight_falls_back_to_uniform():
    results = [
        _result("historical", 0.6, confidence=1.0),
        _result("trend", 0.4, confidence=1.0),
    ]
    # No matching weights for present agents -> uniform fallback.
    weights = {"market": 0.5, "expert": 0.5}
    out = aggregate(results, weights)
    assert out.weights_used == {"historical": 0.5, "trend": 0.5}
    assert out.score == pytest.approx(round((0.6 + 0.4) / 2, 3))


# ---------------------------------------------------------------------------
# Determinism + per-agent passthrough + score validation contract
# ---------------------------------------------------------------------------


def test_aggregate_is_deterministic_across_runs():
    results = [
        _result("historical", 0.62, confidence=0.8),
        _result("trend", 0.31, confidence=0.5),
        _result("risk", 0.55, confidence=0.9, risk_level="low"),
    ]
    weights = {"historical": 0.5, "trend": 0.3, "risk": 0.2}

    a = aggregate(results, weights)
    b = aggregate(results, weights)
    assert a == b


def test_per_agent_scores_passthrough():
    results = [
        _result("historical", 0.62, confidence=0.8),
        _result("trend", 0.31, confidence=0.5),
    ]
    out = aggregate(results, {"historical": 0.5, "trend": 0.5})
    assert out.per_agent == {"historical": 0.62, "trend": 0.31}


def test_confidence_within_unit_interval():
    results = [
        _result("historical", 0.05, confidence=1.0),
        _result("trend", 0.95, confidence=1.0),
    ]
    out = aggregate(results, {"historical": 1.0, "trend": 1.0})
    assert 0.0 <= out.confidence <= 1.0


def test_out_of_range_score_rejected_by_contract():
    """The AgentResult contract guards [0,1]; aggregation relies on it."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AgentResult(agent="historical", score=1.5, confidence=0.5)
    with pytest.raises(ValidationError):
        AgentResult(agent="historical", score=0.5, confidence=-0.1)


def test_agreement_matches_reference_formula():
    """Confidence with two full-confidence agents equals weighted_conf * the
    agreement factor (no conflict), letting us pin the agreement maths."""
    s1, s2 = 0.60, 0.80
    results = [
        _result("historical", s1, confidence=1.0),
        _result("trend", s2, confidence=1.0),
    ]
    out = aggregate(results, {"historical": 1.0, "trend": 1.0})

    spread_std = statistics.pstdev([s1, s2])
    agreement = max(0.0, 1.0 - spread_std * 2.0)
    # weighted_conf = 1.0; spread = 0.20 < 0.35 so no conflict penalty.
    expected = round(min(1.0, 1.0 * (0.6 + 0.4 * agreement)), 3)
    assert out.confidence == expected
    assert out.conflict is False
