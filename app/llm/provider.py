"""Configurable LLM provider (Sprint 04).

Wraps LangChain chat models for OpenAI, Anthropic and Ollama behind one
interface. The
provider is resilient by design (Sprint 04 DoD - LLM errors must never block a
prediction): on missing credentials, timeouts or exceptions it returns a
deterministic, grounded fallback so the pipeline keeps running.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    fallback: bool = False


class LLMProvider:
    """Provider-agnostic chat wrapper with retries and a safe fallback."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.provider = (provider or settings.llm_provider).lower()
        self.model = model or settings.llm_model
        self.timeout = timeout if timeout is not None else settings.llm_timeout_seconds
        self.max_retries = max_retries if max_retries is not None else settings.llm_max_retries
        self._client = None

    # -- client construction -------------------------------------------------
    def _build_client(self):
        if self._client is not None:
            return self._client
        try:
            if self.provider == "openai":
                if not settings.openai_api_key or settings.openai_api_key.startswith("sk-replace"):
                    raise RuntimeError("OpenAI API key not configured")
                from langchain_openai import ChatOpenAI

                self._client = ChatOpenAI(
                    model=self.model,
                    api_key=settings.openai_api_key,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                )
            elif self.provider == "anthropic":
                if (
                    not settings.anthropic_api_key
                    or settings.anthropic_api_key.startswith("sk-replace")
                ):
                    raise RuntimeError("Anthropic API key not configured")
                from langchain_anthropic import ChatAnthropic

                self._client = ChatAnthropic(
                    model=self.model,
                    api_key=settings.anthropic_api_key,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                )
            elif self.provider == "ollama":
                from langchain_ollama import ChatOllama

                # validate_model_on_init pings the server/model at construction, so
                # `available` is only True when Ollama is reachable AND the model exists.
                self._client = ChatOllama(
                    model=self.model,
                    base_url=settings.ollama_base_url,
                    client_kwargs={"timeout": self.timeout},
                    validate_model_on_init=True,
                )
            else:
                raise RuntimeError(f"Unknown LLM provider '{self.provider}'")
        except Exception as exc:
            logger.warning("llm.client.unavailable", provider=self.provider, error=str(exc))
            self._client = None
        return self._client

    @property
    def available(self) -> bool:
        return self._build_client() is not None

    # -- generation ----------------------------------------------------------
    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        client = self._build_client()
        if client is None:
            return self._fallback(prompt)
        try:
            from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

            messages: list[BaseMessage] = []
            if system:
                messages.append(SystemMessage(content=system))
            messages.append(HumanMessage(content=prompt))
            result = client.invoke(messages)
            text = getattr(result, "content", str(result))
            return LLMResponse(text=text, model=self.model, provider=self.provider)
        except Exception as exc:  # timeouts, rate limits, network, etc.
            logger.warning("llm.complete.failed", provider=self.provider, error=str(exc))
            return self._fallback(prompt)

    def _fallback(self, prompt: str) -> LLMResponse:
        """Deterministic, dependency-free response used when the LLM is down."""
        text = (
            "[deterministic-fallback] "
            "Synthesised from the quantified agent contributions; no live LLM was reachable."
        )
        return LLMResponse(text=text, model="fallback", provider=self.provider, fallback=True)


@lru_cache
def get_llm_provider() -> LLMProvider:
    return LLMProvider()
