"""Agent abstraction and execution context (Sprint 02).

Every analytical agent subclasses :class:`BaseAgent` and implements
:meth:`analyze`, returning the normalised :class:`~app.schemas.agent.AgentResult`.
The lifecycle (collect → analyze → normalise) is documented in the Multi-Agent
System note of the vault.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.logging_config import get_logger
from app.schemas.agent import AgentResult

if TYPE_CHECKING:  # pragma: no cover
    from app.connectors.base import BaseConnector
    from app.mcp.manager import MCPManager

logger = get_logger(__name__)


@dataclass
class AgentContext:
    """Execution context handed to each agent by the orchestrator.

    It carries the prediction target plus optional shared resources
    (connectors, MCP manager, free-form context). Agents must treat it as
    read-only.
    """

    entity: str
    domain: str | None = None
    context: dict = field(default_factory=dict)
    connectors: list[BaseConnector] = field(default_factory=list)
    mcp: MCPManager | None = None

    def seed(self, salt: str = "") -> int:
        """Deterministic seed derived from the entity (for reproducible stubs)."""
        digest = hashlib.sha256(f"{self.entity}|{self.domain}|{salt}".encode()).hexdigest()
        return int(digest[:8], 16)


class BaseAgent:
    """Base class for all agents.

    Subclasses set :attr:`name` and implement :meth:`analyze`. The default
    :meth:`run` wraps ``analyze`` with logging and guarantees a valid
    ``AgentResult`` even on failure (so one agent never breaks a prediction).
    """

    #: Unique, stable identifier used for weighting/persistence.
    name: str = "base"
    #: Human description for benchmarking/reporting.
    description: str = ""

    def analyze(self, ctx: AgentContext) -> AgentResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def run(self, ctx: AgentContext) -> AgentResult:
        """Safe wrapper used by the orchestrator."""
        try:
            result = self.analyze(ctx)
            if result.agent != self.name:
                result.agent = self.name
            return result
        except Exception as exc:  # never let one agent break the graph
            logger.warning("agent.failed", agent=self.name, error=str(exc))
            return AgentResult(
                agent=self.name,
                score=0.5,
                confidence=0.0,
                reasoning=f"Agent error, neutral fallback: {exc}",
                extra={"error": True},
            )
