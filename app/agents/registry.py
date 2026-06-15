"""Agent registry — dynamic registration & discovery (Sprint 02).

Adding a new agent is "one class + one registration, zero graph changes"
(Sprint 02 DoD): decorate the class with ``@registry.register``.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.agents.base import BaseAgent
from app.logging_config import get_logger

logger = get_logger(__name__)


class AgentRegistry:
    """Holds the set of available agent classes, keyed by ``name``."""

    def __init__(self) -> None:
        self._agents: dict[str, type[BaseAgent]] = {}

    def register(self, cls: type[BaseAgent]) -> type[BaseAgent]:
        """Class decorator registering an agent under its ``name``."""
        if not getattr(cls, "name", None) or cls.name == "base":
            raise ValueError(f"Agent {cls!r} must define a unique 'name'")
        if cls.name in self._agents:
            logger.warning("agent.registry.override", name=cls.name)
        self._agents[cls.name] = cls
        return cls

    def create(self, name: str) -> BaseAgent:
        return self._agents[name]()

    def names(self) -> list[str]:
        return list(self._agents)

    def all(self, exclude: Iterable[str] = ()) -> list[BaseAgent]:
        excluded = set(exclude)
        return [cls() for name, cls in self._agents.items() if name not in excluded]

    def get(self, name: str) -> type[BaseAgent] | None:
        return self._agents.get(name)


registry = AgentRegistry()
