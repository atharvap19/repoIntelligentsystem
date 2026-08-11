"""LLM provider selection.

The application must not import ``ChatOllama`` in a dozen places, or swapping
providers later means touching a dozen files. Everything goes through
``build_chat_model``, and configuration decides what comes back::

    LLM_PROVIDER=ollama      LLM_MODEL=qwen3:4b        (default)
    LLM_PROVIDER=openai      LLM_MODEL=gpt-...
    LLM_PROVIDER=anthropic   LLM_MODEL=claude-...

Only the Ollama provider is installed. The others raise a clear instruction
rather than a stack trace, because the point of this module is that adding one
is a dependency install plus a registry entry, not a refactor.

Nothing here sends repository content anywhere by default: Ollama is local,
and a remote provider is opt-in through configuration.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "qwen3:4b"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.1


@dataclass(frozen=True)
class LLMSettings:
    """Resolved provider configuration."""

    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    host: str = DEFAULT_HOST
    temperature: float = DEFAULT_TEMPERATURE
    #: Disables qwen3's reasoning trace where the provider supports it.
    reasoning: bool | None = None

    @property
    def is_local(self) -> bool:
        return self.provider == "ollama"

    def describe(self) -> dict:
        """Safe for logs and API responses — never includes credentials."""
        return {
            "provider": self.provider,
            "model": self.model,
            "local": self.is_local,
            "temperature": self.temperature,
        }


def settings_from_env(**overrides: Any) -> LLMSettings:
    """Read provider settings from the environment, with explicit overrides."""
    reasoning_raw = os.getenv("LLM_REASONING")
    reasoning = None
    if reasoning_raw is not None:
        reasoning = reasoning_raw.strip().lower() in {"1", "true", "yes", "on"}

    base = LLMSettings(
        provider=os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower(),
        model=os.getenv("LLM_MODEL", DEFAULT_MODEL).strip(),
        host=os.getenv("LLM_HOST", DEFAULT_HOST).strip(),
        temperature=float(os.getenv("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)),
        reasoning=reasoning,
    )
    if not overrides:
        return base
    return LLMSettings(**{**base.__dict__, **{k: v for k, v in overrides.items() if v is not None}})


def _build_ollama(settings: LLMSettings):
    from langchain_ollama import ChatOllama

    kwargs: dict[str, Any] = {
        "model": settings.model,
        "base_url": settings.host,
        "temperature": settings.temperature,
    }
    if settings.reasoning is not None:
        # qwen3 emits a <think> block by default; disabling it roughly halves
        # generation time at some cost to answer depth.
        kwargs["reasoning"] = settings.reasoning
    return ChatOllama(**kwargs)


def _missing(package: str, provider: str) -> Callable[[LLMSettings], Any]:
    def build(settings: LLMSettings):
        raise RuntimeError(
            f"LLM_PROVIDER={provider!r} needs the {package!r} package. "
            f"Install it and set the provider's API key, or use LLM_PROVIDER=ollama."
        )

    return build


#: Provider name -> factory. Adding a provider is one entry plus its package.
PROVIDERS: dict[str, Callable[[LLMSettings], Any]] = {
    "ollama": _build_ollama,
    "openai": _missing("langchain-openai", "openai"),
    "anthropic": _missing("langchain-anthropic", "anthropic"),
}


def build_chat_model(settings: LLMSettings | None = None, **overrides: Any):
    """Construct the configured chat model."""
    resolved = settings or settings_from_env(**overrides)
    factory = PROVIDERS.get(resolved.provider)
    if factory is None:
        raise ValueError(
            f"Unknown LLM_PROVIDER {resolved.provider!r}. "
            f"Available: {', '.join(sorted(PROVIDERS))}."
        )
    logger.info("Using LLM provider %s", resolved.describe())
    return factory(resolved)
