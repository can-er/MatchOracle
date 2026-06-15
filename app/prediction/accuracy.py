"""Prediction accuracy evaluation (Sprint WC-5).

Compare predicted scorelines/outcomes to the real results and report how good the
model has been so far:

- **outcome accuracy** — got the 1X2 result right (home win / draw / away win),
- **exact-score accuracy** — got the precise scoreline right,
- **mean goal error** — average |predicted - actual| goals over both sides,
- **Brier score** — probabilistic quality of the 1X2 forecast (lower is better).

Pure functions over a list of ``{predicted, p_home_win, p_draw, p_away_win,
actual, entity}`` records, so they're trivially unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AccuracyReport:
    evaluated: int
    outcome_correct: int
    outcome_accuracy: float
    exact_correct: int
    exact_accuracy: float
    mean_goal_error: float
    brier: float
    details: list[dict] = field(default_factory=list)


def _goals(scoreline: str) -> tuple[int, int]:
    home, away = scoreline.split("-")
    return int(home), int(away)


def outcome(home_goals: int, away_goals: int) -> str:
    """1X2 result: 'H' home win, 'D' draw, 'A' away win."""
    if home_goals > away_goals:
        return "H"
    if away_goals > home_goals:
        return "A"
    return "D"


def evaluate(records: list[dict]) -> AccuracyReport:
    """Score predictions against actual results (records with a non-empty actual)."""
    evaluated = 0
    outcome_correct = 0
    exact_correct = 0
    goal_error = 0.0
    brier_total = 0.0
    details: list[dict] = []

    for record in records:
        predicted, actual = record.get("predicted"), record.get("actual")
        if not predicted or not actual:
            continue
        try:
            ph, pa = _goals(predicted)
            ah, aa = _goals(actual)
        except (ValueError, AttributeError):
            continue

        evaluated += 1
        predicted_outcome = outcome(ph, pa)
        actual_outcome = outcome(ah, aa)
        outcome_ok = predicted_outcome == actual_outcome
        exact_ok = (ph, pa) == (ah, aa)
        outcome_correct += int(outcome_ok)
        exact_correct += int(exact_ok)
        goal_error += abs(ph - ah) + abs(pa - aa)

        probs = {
            "H": record.get("p_home_win", 0.0),
            "D": record.get("p_draw", 0.0),
            "A": record.get("p_away_win", 0.0),
        }
        brier_total += sum(
            (probs[o] - (1.0 if o == actual_outcome else 0.0)) ** 2 for o in ("H", "D", "A")
        )

        details.append(
            {
                "entity": record.get("entity"),
                "predicted": predicted,
                "actual": actual,
                "outcome_ok": outcome_ok,
                "exact_ok": exact_ok,
            }
        )

    if evaluated == 0:
        return AccuracyReport(0, 0, 0.0, 0, 0.0, 0.0, 0.0, [])

    return AccuracyReport(
        evaluated=evaluated,
        outcome_correct=outcome_correct,
        outcome_accuracy=round(outcome_correct / evaluated, 3),
        exact_correct=exact_correct,
        exact_accuracy=round(exact_correct / evaluated, 3),
        mean_goal_error=round(goal_error / evaluated, 3),
        brier=round(brier_total / evaluated, 3),
        details=details,
    )
