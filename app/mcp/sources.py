"""MCP data sources (Sprint 07).

A *source* adapts one MCP server (or an in-process demo) into the single signal
the Contextual agent needs: a sentiment snippet for an entity. Two transports:

* ``builtin`` — a deterministic, in-process demo "News Feed MCP". Needs no
  external process, is always available and is used for the Sprint 07 DoD and
  tests. It deliberately **abstains on the live World Cup domain** so demo noise
  never pollutes the flagship predictions.
* ``stdio`` — a real MCP server reached with the official ``mcp`` SDK over a
  stdio transport. Lazy-imported and connected per call (spawn → initialise →
  call tool → close), so a slow or missing binary never blocks startup; any
  failure degrades gracefully to ``None`` and the agent moves on.

Everything here is sync from the caller's point of view: the async SDK round-trip
is bridged by :func:`_run_async`.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
from typing import Protocol, runtime_checkable

from app.logging_config import get_logger

logger = get_logger(__name__)

# Domains whose predictions must stay free of demo/synthetic context noise.
_PROTECTED_DOMAINS = {"worldcup", "world cup", "coupe du monde", "mondial", "wc"}

_POSITIVE = ("win", "strong", "boost", "surge", "confident", "positive", "favour", "favor")
_NEGATIVE = ("loss", "weak", "injury", "doubt", "crisis", "negative", "slump", "concern")


@runtime_checkable
class MCPSource(Protocol):
    """The minimal contract the manager needs from any MCP source."""

    name: str
    transport: str
    role: str
    description: str

    def health(self) -> bool: ...

    def fetch_context(self, entity: str, domain: str | None = None) -> dict | None: ...


def _deterministic_sentiment(entity: str) -> float:
    """Stable, entity-specific sentiment in ``[0, 1]`` (reproducible predictions)."""
    digest = hashlib.sha256(entity.encode("utf-8")).hexdigest()
    return round(int(digest[:8], 16) / 0xFFFFFFFF, 3)


def _sentiment_from_text(text: str) -> float:
    """Map a free-text MCP reply to a sentiment in ``[0, 1]`` (lexicon + fallback)."""
    lowered = text.lower()
    pos = sum(word in lowered for word in _POSITIVE)
    neg = sum(word in lowered for word in _NEGATIVE)
    if pos == neg == 0:
        return _deterministic_sentiment(text)
    return round(0.5 + 0.5 * (pos - neg) / (pos + neg), 3)


def _text_of(result: object) -> str | None:
    """Extract the textual payload from an MCP ``CallToolResult`` defensively."""
    content = getattr(result, "content", None)
    if content is None:
        return None
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts) if parts else None


def _run_async(coro, timeout: float):
    """Run an awaitable to completion from sync code, with a hard timeout.

    Handles both the common case (no running loop) and the rare case of being
    called from within a running loop (offloads to a worker thread).
    """

    async def _guarded():
        return await asyncio.wait_for(coro, timeout)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_guarded())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(_guarded())).result()


class BuiltinNewsSource:
    """In-process demo "News Feed MCP" source.

    Deterministic (hash-based) so tests and predictions stay reproducible. Stands
    in for a real News Feed MCP server without requiring an external process.
    """

    transport = "builtin"

    def __init__(
        self,
        name: str = "demo-news",
        role: str = "contextual",
        description: str = "In-process demo News Feed MCP (deterministic sentiment).",
    ) -> None:
        self.name = name
        self.role = role
        self.description = description

    def health(self) -> bool:
        return True

    def fetch_context(self, entity: str, domain: str | None = None) -> dict | None:
        # Never inject synthetic sentiment into the live flagship (World Cup).
        if (domain or "").lower() in _PROTECTED_DOMAINS:
            return None
        sentiment = _deterministic_sentiment(entity)
        tone = "positive" if sentiment >= 0.5 else "cautious"
        return {
            "source": self.name,
            "sentiment": sentiment,
            "snippet": f"Demo news sentiment for '{entity}' reads {tone} ({sentiment}).",
        }


class StdioMCPSource:
    """A real MCP server reached over stdio via the official ``mcp`` SDK.

    Connects lazily per call so a missing/slow server never blocks startup.
    """

    transport = "stdio"

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        role: str = "contextual",
        tool: str = "sentiment",
        description: str = "",
        timeout: float = 10.0,
    ) -> None:
        self.name = name
        self.command = command
        self.args = list(args or [])
        self.role = role
        self.tool = tool
        self.description = description or f"MCP stdio server '{command}'."
        self.timeout = timeout
        self._healthy: bool | None = None

    def health(self) -> bool:
        # A real round-trip is needed to be sure; treat "configured" as healthy
        # until a fetch proves otherwise.
        return self._healthy is not False

    def fetch_context(self, entity: str, domain: str | None = None) -> dict | None:
        try:
            text = _run_async(self._call_tool(entity, domain), timeout=self.timeout)
        except Exception as exc:  # missing binary, timeout, protocol error…
            self._healthy = False
            logger.warning("mcp.stdio.failed", server=self.name, error=str(exc))
            return None
        if not text:
            return None
        self._healthy = True
        return {
            "source": self.name,
            "sentiment": _sentiment_from_text(text),
            "snippet": text[:280],
        }

    async def _call_tool(self, entity: str, domain: str | None) -> str | None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=self.command, args=self.args)
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool(self.tool, {"entity": entity, "domain": domain})
            return _text_of(result)
