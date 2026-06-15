"""Per-agent accuracy & benchmarking (Sprints 10 & 11).

Pure functions over *observations* — one per agent per resolved prediction:

    {"agent": str, "score": float, "confidence": float,
     "weight": float | None, "label": int | None}

``label`` is the realised binary class (1 = positive / favourable, 0 = negative).
An agent's directional call is ``score >= 0.5``; it *hits* when that matches the
label. From these we derive:

* **accuracy** — directional hit-rate vs reality (feeds the auto-weighting),
* **mean_confidence** + **calibration_error** — does stated confidence match
  realised accuracy? (``|mean_confidence - accuracy|``, lower is better),
* **contribution** — ``mean_weight * accuracy``, a rough "value added" proxy,
* **flag** — ``"underperforming"`` when an agent is worse than a coin flip on a
  meaningful sample (a hook for benchmark-driven re-weighting / disabling).

Being pure (no DB/network) they are trivially unit-tested on a known dataset,
satisfying the Sprint 10 & 11 DoD.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

# Canonical positive / negative class tokens (domain-agnostic engine labels).
_POSITIVE_TOKENS = {"positive", "win", "home", "yes", "up", "bull", "1", "true"}
_NEGATIVE_TOKENS = {"negative", "loss", "lose", "away", "no", "down", "bear", "0", "false"}

# A coin-flip floor and the minimum sample below which we don't flag an agent.
_COIN_FLIP = 0.5
_MIN_FLAG_SAMPLES = 5


@dataclass
class AgentBenchmark:
    agent: str
    samples: int
    accuracy: float
    mean_confidence: float
    calibration_error: float
    mean_weight: float
    contribution: float
    flag: str | None = None


def label_from_actual(actual: str | None) -> int | None:
    """Map a free-text realised outcome to a binary class (or ``None`` if unclear)."""
    if not actual:
        return None
    token = actual.strip().lower()
    if token in _POSITIVE_TOKENS:
        return 1
    if token in _NEGATIVE_TOKENS:
        return 0
    return None


def label_from_scoreline(actual: str | None) -> int | None:
    """Directional label from a ``"home-away"`` scoreline.

    Home win -> 1, away win -> 0, draw -> ``None`` (an agent's directional call
    is undefined for a draw, so those samples are skipped). This is what lets the
    autonomous loop learn from real World Cup results — no human label needed.
    """
    if not actual:
        return None
    try:
        home, away = (int(x) for x in actual.strip().split("-"))
    except (ValueError, AttributeError):
        return None
    if home > away:
        return 1
    if away > home:
        return 0
    return None


def realised_label(actual: str | None) -> int | None:
    """Best realised binary label: a known class token, else a scoreline."""
    token = label_from_actual(actual)
    return token if token is not None else label_from_scoreline(actual)


def _hit(score: float, label: int) -> bool:
    return int(score >= 0.5) == int(label)


def agent_accuracy(observations: list[dict]) -> dict[str, float]:
    """Directional hit-rate per agent over labelled observations."""
    hits: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for obs in observations:
        label = obs.get("label")
        if label is None:
            continue
        agent = obs["agent"]
        total[agent] += 1
        if _hit(float(obs["score"]), int(label)):
            hits[agent] += 1
    return {agent: round(hits[agent] / n, 3) for agent, n in total.items() if n}


def benchmark_agents(observations: list[dict]) -> list[AgentBenchmark]:
    """Full per-agent benchmark report, sorted by accuracy then contribution."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for obs in observations:
        if obs.get("label") is None:
            continue
        grouped[obs["agent"]].append(obs)

    report: list[AgentBenchmark] = []
    for agent, obs in grouped.items():
        n = len(obs)
        accuracy = sum(_hit(float(o["score"]), int(o["label"])) for o in obs) / n
        mean_conf = sum(float(o.get("confidence") or 0.0) for o in obs) / n
        mean_weight = sum(float(o.get("weight") or 0.0) for o in obs) / n
        flag = "underperforming" if n >= _MIN_FLAG_SAMPLES and accuracy < _COIN_FLIP else None
        report.append(
            AgentBenchmark(
                agent=agent,
                samples=n,
                accuracy=round(accuracy, 3),
                mean_confidence=round(mean_conf, 3),
                calibration_error=round(abs(mean_conf - accuracy), 3),
                mean_weight=round(mean_weight, 3),
                contribution=round(mean_weight * accuracy, 3),
                flag=flag,
            )
        )

    report.sort(key=lambda b: (b.accuracy, b.contribution), reverse=True)
    return report
