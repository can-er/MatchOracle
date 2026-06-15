"""Application configuration (Sprint 00).

Centralised, environment-driven settings via pydantic-settings. Every tunable in
``.env.example`` maps to a field here. Import the singleton ``settings``.
"""

from __future__ import annotations

import json
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_WEIGHTS: dict[str, float] = {
    "historical": 0.25,
    "trend": 0.20,
    "contextual": 0.15,
    "risk": 0.15,
    "market": 0.10,
    "expert": 0.15,
}


class Settings(BaseSettings):
    """Typed application settings, prefixed ``MO_`` in the environment."""

    model_config = SettingsConfigDict(
        env_prefix="MO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- General ---
    env: str = "development"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"

    # --- Persistence ---
    # Default is an in-process SQLite DB so the app *imports and boots* even
    # without Postgres; docker compose overrides this with the Postgres URL.
    database_url: str = "sqlite+pysqlite:///./matchoracle.db"
    redis_url: str = "redis://localhost:6379/0"

    # --- LLM ---
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    llm_router_policy: str = "balanced"
    llm_timeout_seconds: int = 30
    llm_max_retries: int = 2

    # --- Orchestration ---
    agent_weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    # --- Auto-weighting (Sprint 10) ---
    autoweight_enabled: bool = True
    autoweight_learning_rate: float = 0.1
    autoweight_min: float = 0.02
    autoweight_max: float = 0.6

    # --- Distributed execution (Sprint 14) ---
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    distributed_agents: bool = False

    # --- Auth / tenancy (Sprint 13) ---
    auth_enabled: bool = False
    jwt_secret: str = "change-me-in-production-with-a-long-random-secret"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    default_tenant: str = "public"
    # Secret backend: "env" (default) or "vault". Vault values come from the env,
    # never the codebase, and only matter when secrets_backend == "vault".
    secrets_backend: str = "env"
    vault_addr: str = ""
    vault_token: str = ""

    # --- MCP (Sprint 07) ---
    mcp_enabled: bool = True
    mcp_config_path: str = "config/mcp_servers.json"

    # --- Observability ---
    metrics_enabled: bool = True

    # --- Connectors (Sprint 08) ---
    openligadb_base_url: str = "https://api.openligadb.de"
    football_league: str = "bl1"  # Bundesliga
    football_season: str = "2025"  # 2025/26 season

    # --- World Cup 2026 (Sprint WC-1) ---
    football_data_base_url: str = "https://api.football-data.org/v4"
    football_data_api_key: str = ""  # free key for football-data.org (live WC form/results)
    wc_competition: str = "WC"

    @field_validator("agent_weights", mode="before")
    @classmethod
    def _parse_weights(cls, value: object) -> object:
        """Accept the weights as a JSON string (from env) or a dict."""
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return dict(DEFAULT_WEIGHTS)
            return json.loads(value)
        return value

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()


settings = get_settings()
