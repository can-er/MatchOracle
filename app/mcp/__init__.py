"""Model Context Protocol integration (Sprint 07).

A small, **sync-friendly** facade over MCP servers so agents can pull contextual
resources without caring about transports or async plumbing. The
:class:`~app.mcp.manager.MCPManager` loads server configs, exposes their status
and answers :meth:`~app.mcp.manager.MCPManager.fetch_context` for the Contextual
agent. See the vault note *Intégration MCP* for the design.
"""

from __future__ import annotations

from app.mcp.manager import MCPManager, get_mcp_manager

__all__ = ["MCPManager", "get_mcp_manager"]
