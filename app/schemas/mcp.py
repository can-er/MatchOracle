"""MCP API schemas (Sprint 07)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MCPServerInfo(BaseModel):
    """One configured MCP source and its current status."""

    name: str = Field(description="Stable identifier of the MCP source.")
    transport: str = Field(description="Transport family: builtin | stdio.")
    role: str = Field(description="What the source feeds, e.g. 'contextual'.")
    description: str = Field(description="Human description of the source.")
    status: str = Field(description="'healthy' or 'unreachable'.")
