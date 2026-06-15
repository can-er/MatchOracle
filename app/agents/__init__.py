"""Multi-agent system (Sprints 02-04).

Importing this package registers all built-in agents via the registry.
"""

# Importing the modules triggers their @registry.register decorators.
from app.agents import (  # noqa: E402,F401
    contextual,
    expert,
    historical,
    market,
    mock_agent,
    risk,
    trend,
)
from app.agents.base import AgentContext, BaseAgent
from app.agents.registry import AgentRegistry, registry

__all__ = ["BaseAgent", "AgentContext", "AgentRegistry", "registry"]
