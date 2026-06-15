"""Risk Assessment Agent (Sprint 04).

Emits a ``risk_level`` plus a ``score``. Higher volatility lowers the score and
the agent's own confidence.
"""

from __future__ import annotations

from app.agents._heuristics import derived_score, signal_from_context
from app.agents._signals import football_matchup, outcome_uncertainty
from app.agents.base import AgentContext, BaseAgent
from app.agents.registry import registry
from app.schemas.agent import AgentResult


def _risk_level(volatility: float) -> str:
    if volatility < 0.34:
        return "low"
    if volatility < 0.67:
        return "medium"
    return "high"


@registry.register
class RiskAgent(BaseAgent):
    """Identifies uncertainty and risk factors."""

    name = "risk"
    description = "Estimates volatility, threat factors and uncertainty."

    def analyze(self, ctx: AgentContext) -> AgentResult:
        # WC-7: a real risk signal — the match-outcome uncertainty from the score
        # model. Backs the stronger side, but with confidence that shrinks as the
        # match gets closer (so the ensemble is less sure about coin-flips).
        matchup = football_matchup(ctx)
        uncertainty = outcome_uncertainty(matchup)
        if matchup is not None and uncertainty is not None:
            volatility, probs = uncertainty
            level = _risk_level(volatility)
            return AgentResult(
                agent=self.name,
                score=round(matchup.rel_strength, 3),
                confidence=round(max(0.25, 0.9 - volatility), 3),
                reasoning=(
                    f"Real data ({matchup.source}): outcome uncertainty {volatility} "
                    f"→ risk '{level}' (H/D/A {probs['home']}/{probs['draw']}/{probs['away']})."
                ),
                extra={"source": matchup.source, "risk_level": level, "uncertainty": volatility},
            )

        hint = signal_from_context(ctx, ["volatility", "threat_level"])

        # A connector is wired but the matchup didn't resolve and there's no context
        # hint -> abstain rather than inject heuristic noise.
        if hint is None and ctx.connectors:
            return AgentResult(
                agent=self.name,
                score=0.5,
                confidence=0.2,
                reasoning=f"No real risk signal for '{ctx.entity}' yet — abstaining (neutral).",
                extra={"abstained": True, "risk_level": "unknown"},
            )

        volatility = derived_score(ctx, self.name + "::vol", low=0.1, high=0.95)
        if hint is not None:
            volatility = round(0.5 * volatility + 0.5 * hint, 3)

        level = _risk_level(volatility)
        # A favourable (high) score means *low* risk.
        score = round(1.0 - volatility, 3)
        # Confidence shrinks as volatility rises.
        confidence = round(max(0.2, 0.9 - volatility * 0.5), 3)
        return AgentResult(
            agent=self.name,
            score=score,
            confidence=confidence,
            reasoning=(
                f"Estimated volatility {volatility} → risk level '{level}' for '{ctx.entity}'."
            ),
            extra={"risk_level": level, "volatility": volatility},
        )
