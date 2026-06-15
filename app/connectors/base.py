"""Connector abstraction (Sprint 08).

A connector adapts an external system (REST API, DB, MCP…) into signals the
agents can consume. Concrete connectors set :attr:`name`/:attr:`type` and
implement :meth:`health`; domain-specific data methods are added by subclasses.
"""

from __future__ import annotations


class BaseConnector:
    """Base class for all data connectors."""

    #: Stable identifier (matches the persisted ``connectors.name``).
    name: str = "base"
    #: Connector family: rest | graphql | sql | nosql | mcp.
    type: str = "rest"
    #: Optional domain hint (e.g. "sports") used to select a connector.
    domain: str | None = None

    def health(self) -> bool:
        """Return True if the underlying source is reachable."""
        raise NotImplementedError
