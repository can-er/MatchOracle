"""Secret resolution (Sprint 13).

Secrets are sourced from the environment by default and, optionally, from
HashiCorp Vault — never hard-coded. The Vault backend is lazy (``hvac`` imported
only when selected) so the app has no hard runtime dependency on Vault; a dev
Vault container is provided in docker-compose for the enterprise posture.
"""

from __future__ import annotations

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class EnvSecretProvider:
    """Resolves secrets straight from the typed settings / environment."""

    def get(self, name: str, default: str | None = None) -> str | None:
        return getattr(settings, name, None) or default


class VaultSecretProvider:
    """Resolves secrets from a Vault KV v2 mount (lazy ``hvac``)."""

    def __init__(self, addr: str, token: str, mount: str = "secret", path: str = "matchoracle"):
        self.addr, self.token, self.mount, self.path = addr, token, mount, path
        self._cache: dict[str, str] | None = None

    def _load(self) -> dict[str, str]:
        if self._cache is not None:
            return self._cache
        try:
            import hvac  # lazy: only needed when the Vault backend is selected

            client = hvac.Client(url=self.addr, token=self.token)
            resp = client.secrets.kv.v2.read_secret_version(path=self.path, mount_point=self.mount)
            self._cache = dict(resp["data"]["data"])
        except Exception as exc:  # Vault down / missing path -> fall back to env
            logger.warning("secrets.vault.unavailable", error=str(exc))
            self._cache = {}
        return self._cache

    def get(self, name: str, default: str | None = None) -> str | None:
        return self._load().get(name) or getattr(settings, name, None) or default


def get_secret_provider():
    """Pick the configured secret backend (``env`` by default)."""
    if settings.secrets_backend.lower() == "vault" and settings.vault_addr and settings.vault_token:
        return VaultSecretProvider(settings.vault_addr, settings.vault_token)
    return EnvSecretProvider()


def jwt_secret() -> str:
    """The active JWT signing secret (from the secret backend, never literal)."""
    return get_secret_provider().get("jwt_secret") or settings.jwt_secret
