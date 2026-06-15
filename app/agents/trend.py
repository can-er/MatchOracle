"""Trend Analysis Agent (Sprint 03)."""

from __future__ import annotations

from app.agents._heuristics import derived_confidence, derived_score, signal_from_context
from app.agents._signals import football_matchup
from app.agents.base import AgentContext, BaseAgent
from app.agents.registry import registry
from app.schemas.agent import AgentResult


@registry.register
class TrendAgent(BaseAgent):
    """Captures recent momentum and emerging short-term patterns."""

    name = "trend"
    description = "Analyses recent events, short-term performance and momentum."

    def analyze(self, ctx: AgentContext) -> AgentResult:
        m = football_matchup(ctx)
        if m is not None and m.rel_momentum is not None:
            home = m.home
            if m.away is not None and m.away.momentum is not None:
                reasoning = (
                    f"Real data ({m.source}): {home.team} form {home.form or 'n/a'} vs "
                    f"{m.away.team} {m.away.form or 'n/a'} → momentum edge {m.rel_momentum:.2f}."
                )
            else:
                reasoning = (
                    f"Real data ({m.source}): {home.team} recent form "
                    f"{home.form or 'n/a'} → momentum {m.rel_momentum:.2f}."
                )
            return AgentResult(
                agent=self.name,
                score=m.rel_momentum,
                confidence=0.8,
                reasoning=reasoning,
                extra={
                    "source": m.source,
                    "home": home.team,
                    "away": m.away.team if m.away else None,
                    "rel_momentum": m.rel_momentum,
                },
            )

        score = derived_score(ctx, self.name, low=0.25, high=0.9)
        hint = signal_from_context(ctx, ["recent_form", "momentum"])
        if hint is not None:
            score = round(0.4 * score + 0.6 * hint, 3)
        return AgentResult(
            agent=self.name,
            score=score,
            confidence=derived_confidence(ctx, self.name, low=0.55, high=0.85),
            reasoning=(
                f"Recent momentum for '{ctx.entity}' evaluated to {score}. "
                "Weighted toward the latest short-term signals."
            ),
        )
