"""Tests for the prediction accuracy evaluator (Sprint WC-5)."""

from __future__ import annotations

from app.prediction.accuracy import evaluate, outcome


def _rec(entity, predicted, actual, ph=0.5, pd=0.3, pa=0.2):
    return {
        "entity": entity,
        "predicted": predicted,
        "actual": actual,
        "p_home_win": ph,
        "p_draw": pd,
        "p_away_win": pa,
    }


def test_outcome_1x2() -> None:
    assert outcome(2, 0) == "H"
    assert outcome(1, 1) == "D"
    assert outcome(0, 2) == "A"


def test_evaluate_counts_outcomes_exact_and_goal_error() -> None:
    records = [
        _rec("A", "2-0", "2-0"),  # exact + outcome
        _rec("B", "2-1", "3-1"),  # outcome ok, not exact (goal err = 1)
        _rec("C", "1-0", "0-1"),  # wrong outcome (goal err = 2)
    ]
    report = evaluate(records)
    assert report.evaluated == 3
    assert report.outcome_correct == 2
    assert report.outcome_accuracy == round(2 / 3, 3)
    assert report.exact_correct == 1
    assert report.exact_accuracy == round(1 / 3, 3)
    assert report.mean_goal_error == 1.0  # (0 + 1 + 2) / 3
    assert len(report.details) == 3


def test_evaluate_ignores_records_without_an_actual() -> None:
    assert evaluate([]).evaluated == 0
    assert evaluate([_rec("X", "2-0", None)]).evaluated == 0


def test_brier_rewards_confident_correct_and_punishes_confident_wrong() -> None:
    perfect = evaluate([_rec("A", "2-0", "2-0", ph=1.0, pd=0.0, pa=0.0)])
    assert perfect.brier == 0.0
    # Sure of an away win (pa=1) but home won -> (0-1)^2 + 0 + (1-0)^2 = 2.0
    worst = evaluate([_rec("A", "0-2", "2-0", ph=0.0, pd=0.0, pa=1.0)])
    assert worst.brier == 2.0
