"""MCP management endpoints (Sprint 07).

Lets an admin see which MCP servers are wired in and whether they're reachable
(user story 07-3). Source execution lives in the agents; this is read-only.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.config import settings
from app.mcp import get_mcp_manager
from app.schemas.mcp import MCPServerInfo

router = APIRouter(tags=["mcp"])


@router.get(
    "/mcp/servers",
    response_model=list[MCPServerInfo],
    summary="List configured MCP servers and their status",
)
def list_mcp_servers() -> list[MCPServerInfo]:
    if not settings.mcp_enabled:
        return []
    return [MCPServerInfo(**server) for server in get_mcp_manager().servers()]
