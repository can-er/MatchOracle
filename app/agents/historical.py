"""Historical Analysis Agent (Sprint 03)."""

from __future__ import annotations

from app.agents._heuristics import derived_confidence, derived_score, signal_from_context
from app.agents._signals import football_matchup
from app.agents.base import AgentContext, BaseAgent
from app.agents.registry import registry
from app.schemas.agent import AgentResult


@registry.register
class HistoricalAgent(BaseAgent):
    """Estimates a baseline inclination from long-term historical data."""

    name = "historical"
    description = "Analyses historical events, past outcomes and long-term performance."

    def analyze(self, ctx: AgentContext) -> AgentResult:
        m = football_matchup(ctx)
        if m is not None:
            home = m.home
            if m.away is not None:
                reasoning = (
                    f"Real data ({m.source}): {home.team} win rate {home.win_rate:.0%} "
                    f"vs {m.away.team} {m.away.win_rate:.0%} → edge {m.rel_win_rate:.2f}."
                )
            else:
                reasoning = (
                    f"Real data ({m.source}): {home.team} win rate {home.win_rate:.0%} "
                    f"(rank #{home.rank})."
                )
            return AgentResult(
                agent=self.name,
                score=m.rel_win_rate,
                confidence=round(min(0.95, 0.6 + home.played / 70), 3),
                reasoning=reasoning,
                extra={
                    "source": m.source,
                    "home": home.team,
                    "away": m.away.team if m.away else None,
                    "rel_win_rate": m.rel_win_rate,
                },
            )

        score = derived_score(ctx, self.name, low=0.3, high=0.85)
        hint = signal_from_context(ctx, ["historical_winrate", "past_performance"])
        if hint is not None:
            score = round(0.5 * score + 0.5 * hint, 3)
        return AgentResult(
            agent=self.name,
            score=score,
            confidence=derived_confidence(ctx, self.name, low=0.6, high=0.9),
            reasoning=(
                f"Long-term historical signal for '{ctx.entity}' evaluated to {score}. "
                "Based on archived outcomes and historical statistics."
            ),
        )
