"""Expert LLM Agent (Sprint 04).

Provides qualitative reasoning. It reads the other agents' results (injected by
the orchestrator as ``ctx.context['peer_results']``) and asks the LLM (via the
multi-model router) for a recommendation + explanation. If the LLM is
unavailable it synthesises a grounded explanation from the quantified peer
contributions — so a prediction is always explainable.
"""

from __future__ import annotations

from app.agents.base import AgentContext, BaseAgent
from app.agents.registry import registry
from app.llm.router import get_model_router
from app.schemas.agent import AgentResult

SYSTEM_PROMPT = (
    "You are the Expert agent in a multi-agent prediction platform. "
    "Given the quantified outputs of specialist agents, produce a concise, "
    "grounded explanation (2-3 sentences) of the likely outcome. "
    "Never invent facts beyond the provided signals."
)


def _peer_summary(peers: list[dict]) -> str:
    parts = [f"{p.get('agent')}={p.get('score'):.2f}(c={p.get('confidence'):.2f})" for p in peers]
    return ", ".join(parts)


@registry.register
class ExpertAgent(BaseAgent):
    name = "expert"
    description = "Qualitative reasoning and natural-language explanation via an LLM."

    def analyze(self, ctx: AgentContext) -> AgentResult:
        peers: list[dict] = ctx.context.get("peer_results", [])
        if peers:
            avg = sum(p.get("score", 0.5) for p in peers) / len(peers)
        else:
            avg = ctx.seed("expert") % 1000 / 1000.0
        recommendation = "positive" if avg >= 0.5 else "negative"

        # When the real-data agents identified the two sides, decide the favoured
        # side deterministically and ask the LLM to *justify* it (not re-derive it
        # from the number — small models get that backwards).
        home_team = away_team = None
        for peer in peers:
            extra = peer.get("extra") or {}
            if extra.get("home"):
                home_team, away_team = extra.get("home"), extra.get("away")
                break

        favoured: str | None = None
        if home_team and away_team:
            favoured = home_team if avg >= 0.5 else away_team
            guidance = (
                f"The aggregated signals favour {favoured}. In 2-3 sentences, explain "
                f"why {favoured} is the more likely winner, grounded ONLY in the signals."
            )
        else:
            guidance = "Explain the likely outcome and justify it from the signals above."

        prompt = (
            f"Entity: {ctx.entity}\n"
            f"Domain: {ctx.domain or 'generic'}\n"
            f"Agent signals: {_peer_summary(peers) if peers else 'none'}\n"
            f"Aggregate inclination (first team's perspective): {avg:.2f}\n"
            f"{guidance}"
        )
        router = get_model_router()
        complexity = 0.7 if len(peers) >= 4 else 0.4
        llm = router.complete(prompt, system=SYSTEM_PROMPT, complexity=complexity)

        if llm.fallback:
            explanation = (
                f"Aggregate agent inclination for '{ctx.entity}' is {avg:.2f} "
                f"({recommendation}). Signals: {_peer_summary(peers) if peers else 'n/a'}."
            )
        else:
            explanation = llm.text.strip()

        return AgentResult(
            agent=self.name,
            score=round(avg, 3),
            confidence=0.78 if not llm.fallback else 0.6,
            reasoning=explanation,
            extra={
                "recommendation": recommendation,
                "favoured": favoured,
                "explanation": explanation,
                "llm_model": llm.model,
                "llm_fallback": llm.fallback,
            },
        )
