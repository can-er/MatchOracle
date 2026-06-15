"""Contextual Agent (Sprint 03, MCP-aware in Sprint 07).

Consumes environmental/contextual signals. When an MCP manager is present it
pulls a contextual resource (e.g. a News Feed MCP) to enrich the analysis.
"""

from __future__ import annotations

from app.agents._heuristics import derived_confidence, derived_score, signal_from_context
from app.agents._signals import football_matchup, host_advantage
from app.agents.base import AgentContext, BaseAgent
from app.agents.registry import registry
from app.logging_config import get_logger
from app.schemas.agent import AgentResult

logger = get_logger(__name__)


@registry.register
class ContextualAgent(BaseAgent):
    """Analyses news, external events and environmental context."""

    name = "contextual"
    description = "Analyses news articles, external events and business context."

    def analyze(self, ctx: AgentContext) -> AgentResult:
        sources: list[str] = []

        # WC-7: a real, domain-specific factor outranks any generic feed —
        # World Cup 2026 host-nation advantage (USA / Canada / Mexico play at home).
        host = host_advantage(football_matchup(ctx))
        if host is not None:
            host_score, host_team = host
            return AgentResult(
                agent=self.name,
                score=round(host_score, 3),
                confidence=0.6,
                reasoning=f"Host nation {host_team} enjoys home advantage at WC 2026.",
                extra={"source": "host_advantage", "host": host_team, "sources": sources},
            )

        real: float | None = None

        # Sprint 07: consume an MCP contextual resource if one is wired in. The
        # demo source abstains on the World Cup domain, so the live flagship is
        # never fed synthetic sentiment.
        if ctx.mcp is not None:
            try:
                snippet = ctx.mcp.fetch_context(ctx.entity, ctx.domain)
                if snippet:
                    sources.append(snippet.get("source", "mcp"))
                    nudge = snippet.get("sentiment")
                    if isinstance(nudge, (int, float)):
                        real = float(nudge)
            except Exception as exc:  # MCP must never break a prediction
                logger.warning("contextual.mcp.failed", error=str(exc))

        hint = signal_from_context(ctx, ["news_sentiment", "context_score"])
        if hint is not None:
            real = hint if real is None else round(0.5 * real + 0.5 * hint, 3)

        # A connector is wired but no real contextual factor applies -> abstain
        # rather than inject heuristic noise (news/sentiment feed = future plug-in).
        if real is None and ctx.connectors:
            return AgentResult(
                agent=self.name,
                score=0.5,
                confidence=0.2,
                reasoning=f"No real contextual signal for '{ctx.entity}' — abstaining.",
                extra={"abstained": True, "sources": sources},
            )

        score = (
            round(real, 3) if real is not None else derived_score(ctx, self.name, low=0.2, high=0.8)
        )
        return AgentResult(
            agent=self.name,
            score=score,
            confidence=derived_confidence(ctx, self.name, low=0.45, high=0.8),
            reasoning=f"Contextual/environmental signal for '{ctx.entity}' evaluated to {score}.",
            extra={"sources": sources},
        )
