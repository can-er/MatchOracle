"""MCP integration tests (Sprint 07).

Covers the source abstractions, the manager (config loading + graceful
degradation), the Contextual agent consuming an MCP resource, and the
``/mcp/servers`` admin endpoint — all offline, no real MCP server required.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.agents.base import AgentContext
from app.agents.registry import registry
from app.main import app
from app.mcp.manager import MCPManager, _source_from_config
from app.mcp.sources import (
    BuiltinNewsSource,
    StdioMCPSource,
    _sentiment_from_text,
    _text_of,
)


# --------------------------------------------------------------------------- #
# Builtin demo source
# --------------------------------------------------------------------------- #
def test_builtin_source_is_deterministic() -> None:
    src = BuiltinNewsSource()
    a = src.fetch_context("Acme Corp", "finance")
    b = src.fetch_context("Acme Corp", "finance")
    assert a == b
    assert a is not None
    assert 0.0 <= a["sentiment"] <= 1.0
    assert a["source"] == "demo-news"
    assert src.health() is True


def test_builtin_source_abstains_on_worldcup() -> None:
    """The demo source must never feed synthetic sentiment to the live flagship."""
    src = BuiltinNewsSource()
    assert src.fetch_context("France vs Brazil", "worldcup") is None
    assert src.fetch_context("Mexico vs South Africa", "Coupe du Monde") is None
    # Non-protected domains still get a snippet.
    assert src.fetch_context("France vs Brazil", "sports") is not None


# --------------------------------------------------------------------------- #
# Manager: config loading + degradation
# --------------------------------------------------------------------------- #
def test_manager_lists_servers_with_status() -> None:
    manager = MCPManager(sources=[BuiltinNewsSource()])
    servers = manager.servers()
    assert len(servers) == 1
    assert servers[0]["name"] == "demo-news"
    assert servers[0]["transport"] == "builtin"
    assert servers[0]["status"] == "healthy"


def test_manager_fetch_context_skips_non_contextual_roles() -> None:
    market_only = BuiltinNewsSource(name="market-feed", role="market")
    manager = MCPManager(sources=[market_only])
    assert manager.fetch_context("Acme", "finance") is None


def test_manager_survives_a_raising_source() -> None:
    class _Boom:
        name = "boom"
        transport = "builtin"
        role = "contextual"
        description = "always fails"

        def health(self) -> bool:
            return True

        def fetch_context(self, entity, domain=None):
            raise RuntimeError("nope")

    manager = MCPManager(sources=[_Boom(), BuiltinNewsSource()])
    # The bad source is skipped; the good one answers.
    snippet = manager.fetch_context("Acme", "finance")
    assert snippet is not None
    assert snippet["source"] == "demo-news"


def test_source_from_config_variants() -> None:
    assert isinstance(_source_from_config({"transport": "builtin", "name": "n"}), BuiltinNewsSource)
    stdio = _source_from_config(
        {"transport": "stdio", "name": "ext", "command": "python", "args": ["-m", "srv"]}
    )
    assert isinstance(stdio, StdioMCPSource)
    # stdio with no command -> rejected; unknown transport -> rejected.
    assert _source_from_config({"transport": "stdio", "name": "bad"}) is None
    assert _source_from_config({"transport": "carrier-pigeon", "name": "x"}) is None


# --------------------------------------------------------------------------- #
# Stdio source helpers (parsing) + graceful failure
# --------------------------------------------------------------------------- #
def test_sentiment_from_text_lexicon() -> None:
    assert _sentiment_from_text("strong win, confident surge") > 0.5
    assert _sentiment_from_text("injury crisis, doubt and slump") < 0.5
    # Neutral text falls back to a stable deterministic value in range.
    val = _sentiment_from_text("the match is on tuesday")
    assert 0.0 <= val <= 1.0


def test_text_of_extracts_content_blocks() -> None:
    class _Block:
        def __init__(self, text):
            self.text = text

    class _Result:
        content = [_Block("hello"), _Block("  "), _Block("world")]

    assert _text_of(_Result()) == "hello\nworld"
    assert _text_of(object()) is None


def test_stdio_source_degrades_to_none_when_unreachable() -> None:
    # No such binary -> the async round-trip fails -> None, and health flips.
    src = StdioMCPSource(name="missing", command="definitely-not-a-real-binary-xyz", timeout=2.0)
    assert src.fetch_context("Acme", "finance") is None
    assert src.health() is False


def test_stdio_source_maps_text_to_snippet(monkeypatch) -> None:
    src = StdioMCPSource(name="fake", command="x")

    def _fake_run(coro, timeout):
        coro.close()  # we won't await the real round-trip in this unit test
        return "strong winning form"

    monkeypatch.setattr("app.mcp.sources._run_async", _fake_run)
    snippet = src.fetch_context("Acme", "finance")
    assert snippet is not None
    assert snippet["source"] == "fake"
    assert snippet["sentiment"] > 0.5
    assert src.health() is True


# --------------------------------------------------------------------------- #
# Contextual agent consumes an MCP resource (Sprint 07 DoD)
# --------------------------------------------------------------------------- #
def test_contextual_agent_consumes_mcp_resource() -> None:
    manager = MCPManager(sources=[BuiltinNewsSource()])
    ctx = AgentContext(entity="Acme Corp", domain="finance", mcp=manager)
    result = registry.create("contextual").run(ctx)

    expected = BuiltinNewsSource().fetch_context("Acme Corp", "finance")
    assert result.score == round(expected["sentiment"], 3)
    assert "demo-news" in result.extra["sources"]


def test_contextual_agent_ignores_mcp_on_worldcup_host_match() -> None:
    """Real host-advantage factor outranks MCP, and the demo abstains on WC."""
    from app.connectors.worldcup import WorldCupConnector

    manager = MCPManager(sources=[BuiltinNewsSource()])
    ctx = AgentContext(
        entity="Mexico vs South Africa",
        domain="worldcup",
        connectors=[WorldCupConnector()],
        mcp=manager,
    )
    result = registry.create("contextual").run(ctx)
    assert result.extra.get("source") == "host_advantage"
    assert result.extra.get("host") == "Mexico"


def test_contextual_agent_survives_broken_mcp() -> None:
    class _Boom:
        name = "boom"
        transport = "builtin"
        role = "contextual"
        description = ""

        def health(self):
            return True

        def fetch_context(self, entity, domain=None):
            raise RuntimeError("mcp down")

    ctx = AgentContext(entity="Acme", domain="finance", mcp=MCPManager(sources=[_Boom()]))
    result = registry.create("contextual").run(ctx)
    # No crash; a valid result is still produced.
    assert 0.0 <= result.score <= 1.0


# --------------------------------------------------------------------------- #
# Admin endpoint
# --------------------------------------------------------------------------- #
def test_mcp_servers_endpoint_lists_demo() -> None:
    client = TestClient(app)
    resp = client.get("/api/v1/mcp/servers")
    assert resp.status_code == 200
    body = resp.json()
    assert any(s["name"] == "demo-news" and s["transport"] == "builtin" for s in body)
