"""MCP manager (Sprint 07).

Owns the configured set of MCP sources, reports their status and answers the one
question agents ask: *what's the contextual signal for this entity?* Config is
JSON at ``settings.mcp_config_path``; if it's missing or invalid the manager
still comes up with the built-in demo source so MCP is live out of the box.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.config import settings
from app.logging_config import get_logger
from app.mcp.sources import BuiltinNewsSource, MCPSource, StdioMCPSource

logger = get_logger(__name__)


class MCPManager:
    """Facade over the configured MCP sources."""

    def __init__(self, sources: list[MCPSource] | None = None) -> None:
        self._sources: list[MCPSource] = sources if sources is not None else _load_sources()

    @property
    def sources(self) -> list[MCPSource]:
        return list(self._sources)

    def servers(self) -> list[dict]:
        """Describe each configured source + a best-effort health status."""
        described: list[dict] = []
        for source in self._sources:
            try:
                healthy = source.health()
            except Exception:  # health probes must never raise to the caller
                healthy = False
            described.append(
                {
                    "name": source.name,
                    "transport": source.transport,
                    "role": source.role,
                    "description": source.description,
                    "status": "healthy" if healthy else "unreachable",
                }
            )
        return described

    def fetch_context(self, entity: str, domain: str | None = None) -> dict | None:
        """Return the first contextual snippet a source yields (or ``None``)."""
        for source in self._sources:
            if getattr(source, "role", "contextual") != "contextual":
                continue
            try:
                snippet = source.fetch_context(entity, domain)
            except Exception as exc:  # one bad source must not break a prediction
                logger.warning("mcp.source.failed", server=source.name, error=str(exc))
                continue
            if snippet:
                return snippet
        return None


def _source_from_config(entry: dict) -> MCPSource | None:
    transport = (entry.get("transport") or "builtin").lower()
    name = entry.get("name") or transport
    role = entry.get("role", "contextual")
    description = entry.get("description", "")

    if transport == "builtin":
        return BuiltinNewsSource(
            name=name,
            role=role,
            description=description or "In-process demo News Feed MCP.",
        )
    if transport == "stdio":
        command = entry.get("command")
        if not command:
            logger.warning("mcp.config.stdio.no_command", name=name)
            return None
        return StdioMCPSource(
            name=name,
            command=command,
            args=entry.get("args", []),
            role=role,
            tool=entry.get("tool", "sentiment"),
            description=description,
            timeout=float(entry.get("timeout", 10.0)),
        )
    logger.warning("mcp.config.unknown_transport", name=name, transport=transport)
    return None


def _load_sources() -> list[MCPSource]:
    if not settings.mcp_enabled:
        return []

    path = Path(settings.mcp_config_path)
    if not path.exists():
        return [BuiltinNewsSource()]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("mcp.config.invalid", path=str(path), error=str(exc))
        return [BuiltinNewsSource()]

    sources = [s for entry in raw.get("servers", []) if (s := _source_from_config(entry))]
    if not sources:
        return [BuiltinNewsSource()]
    logger.info("mcp.sources.loaded", count=len(sources), names=[s.name for s in sources])
    return sources


@lru_cache
def get_mcp_manager() -> MCPManager:
    """Return the cached process-wide MCP manager."""
    return MCPManager()
