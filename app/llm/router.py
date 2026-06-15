"""Multi-model router (Sprint 12).

Routes a request to a model tier according to a configurable policy
(``cost`` | ``quality`` | ``balanced``) and the request's complexity. Keeps a
simple, explainable policy first (per the sprint's risk mitigation).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.config import settings
from app.llm.provider import LLMProvider, LLMResponse
from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ModelTier:
    name: str
    model: str
    cost: float  # relative cost per call
    quality: float  # relative quality score


# Tiers are illustrative; real model IDs are environment-driven.
DEFAULT_TIERS = [
    ModelTier(name="small", model="gpt-4o-mini", cost=0.2, quality=0.7),
    ModelTier(name="large", model="gpt-4o", cost=1.0, quality=0.95),
]


class ModelRouter:
    """Selects a model tier and delegates to a :class:`LLMProvider`."""

    def __init__(self, policy: str | None = None, tiers: list[ModelTier] | None = None) -> None:
        self.policy = (policy or settings.llm_router_policy).lower()
        self.tiers = tiers or DEFAULT_TIERS
        self._providers: dict[str, LLMProvider] = {}

    def select(self, *, complexity: float = 0.5) -> ModelTier:
        """Pick a tier. ``complexity`` ∈ [0,1] escalates to higher tiers."""
        ordered = sorted(self.tiers, key=lambda t: t.cost)
        if self.policy == "cost":
            chosen = ordered[0]
        elif self.policy == "quality":
            chosen = max(self.tiers, key=lambda t: t.quality)
        else:  # balanced: escalate when the task is complex
            chosen = ordered[-1] if complexity >= 0.6 and len(ordered) > 1 else ordered[0]
        logger.debug("llm.router.select", policy=self.policy, tier=chosen.name)
        return chosen

    def _effective_model(self, tier: ModelTier) -> str:
        """Resolve the tier to a model valid for the active provider.

        The default tiers carry illustrative OpenAI model IDs. When a different
        provider is configured (e.g. Ollama), the tier names don't exist there,
        so route to the env-configured ``llm_model`` instead.
        """
        if settings.llm_provider.lower() == "openai":
            return tier.model
        return settings.llm_model

    def _provider_for(self, tier: ModelTier) -> LLMProvider:
        model = self._effective_model(tier)
        if model not in self._providers:
            self._providers[model] = LLMProvider(model=model)
        return self._providers[model]

    def complete(
        self, prompt: str, *, system: str | None = None, complexity: float = 0.5
    ) -> LLMResponse:
        tier = self.select(complexity=complexity)
        return self._provider_for(tier).complete(prompt, system=system)


@lru_cache
def get_model_router() -> ModelRouter:
    return ModelRouter()
