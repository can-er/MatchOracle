"""Human-in-the-loop feedback → reward signal (Sprint 12).

Pure helpers turning a human verdict on a prediction into:

* a **reward** (approve ``+1`` / reject ``-1`` / correct ``+0.5`` when a
  correction is supplied, else ``0``), stored on the feedback row, and
* a **realised binary label** the closed feedback loop can learn from — so a
  human correction reshapes future agent weights exactly like a real outcome
  does (Sprint 12 DoD: "a human correction changes future behaviour").

Reusing :func:`~app.prediction.benchmark.label_from_actual` keeps the label
semantics identical to the outcome-tracking path (Sprint 10).
"""

from __future__ import annotations

from app.prediction.benchmark import label_from_actual

APPROVE, REJECT, CORRECT = "approve", "reject", "correct"
VERDICTS = frozenset({APPROVE, REJECT, CORRECT})


def feedback_reward(verdict: str, corrected: str | None = None) -> float:
    """Scalar reward for a verdict (bounded, used as the RL signal)."""
    v = verdict.strip().lower()
    if v == APPROVE:
        return 1.0
    if v == REJECT:
        return -1.0
    if v == CORRECT:
        return 0.5 if corrected else 0.0
    return 0.0


def label_from_feedback(
    verdict: str, predicted: str, corrected: str | None = None
) -> int | None:
    """Realised binary class implied by a human verdict (or ``None`` if unclear)."""
    v = verdict.strip().lower()
    predicted_label = label_from_actual(predicted)
    if v == APPROVE:
        return predicted_label
    if v == REJECT:
        return None if predicted_label is None else 1 - predicted_label
    if v == CORRECT:
        return label_from_actual(corrected)
    return None
