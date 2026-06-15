"""Mock agent (Sprint 02) — validates the end-to-end flow without dependencies."""

from __future__ import annotations

from app.agents.base import AgentContext, BaseAgent
from app.agents.registry import registry
from app.schemas.agent import AgentResult


@registry.register
class MockAgent(BaseAgent):
    name = "mock"
    description = "Deterministic mock agent used to validate the orchestration flow."

    def analyze(self, ctx: AgentContext) -> AgentResult:
        seed = ctx.seed("mock")
        score = (seed % 1000) / 1000.0
        return AgentResult(
            agent=self.name,
            score=round(score, 3),
            confidence=0.5,
            reasoning="Mock deterministic output derived from the entity hash.",
        )
