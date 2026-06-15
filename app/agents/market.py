"""Market Intelligence Agent (Sprint 04)."""

from __future__ import annotations

from app.agents._heuristics import derived_confidence, derived_score, signal_from_context
from app.agents._signals import football_matchup
from app.agents.base import AgentContext, BaseAgent
from app.agents.registry import registry
from app.schemas.agent import AgentResult


@registry.register
class MarketAgent(BaseAgent):
    """Analyses external signals and collective sentiment."""

    name = "market"
    description = "Analyses market movements, public sentiment and consensus forecasts."

    def analyze(self, ctx: AgentContext) -> AgentResult:
        m = football_matchup(ctx)
        if m is not None:
            home = m.home
            if m.away is not None:
                reasoning = (
                    f"Real data ({m.source}): {home.team} #{home.rank} vs "
                    f"{m.away.team} #{m.away.rank} → strength edge {m.rel_strength:.2f}."
                )
            else:
                reasoning = (
                    f"Real data ({m.source}): {home.team} ranked #{home.rank} "
                    f"→ strength {m.rel_strength:.2f}."
                )
            return AgentResult(
                agent=self.name,
                score=m.rel_strength,
                confidence=0.78,
                reasoning=reasoning,
                extra={
                    "source": m.source,
                    "home": home.team,
                    "away": m.away.team if m.away else None,
                    "rel_strength": m.rel_strength,
                },
            )

        score = derived_score(ctx, self.name, low=0.3, high=0.88)
        hint = signal_from_context(ctx, ["market_sentiment", "consensus", "odds"])
        if hint is not None:
            score = round(0.45 * score + 0.55 * hint, 3)
        return AgentResult(
            agent=self.name,
            score=score,
            confidence=derived_confidence(ctx, self.name, low=0.5, high=0.82),
            reasoning=(
                f"External market/sentiment signal for '{ctx.entity}' evaluated to {score}."
            ),
        )
