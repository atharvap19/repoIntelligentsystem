"""LLM provider selection.

The application must not import ``ChatGoogleGenerativeAI`` in a dozen places,
or swapping providers later means touching a dozen files. Everything goes
through ``build_chat_model``, and configuration decides what comes back::

    LLM_PROVIDER=gemini      LLM_MODEL=gemini-3.6-flash    (default)
    LLM_PROVIDER=openai      LLM_MODEL=gpt-...
    LLM_PROVIDER=anthropic   LLM_MODEL=claude-...

Only the Gemini provider is installed. The others raise a clear instruction
rather than a stack trace, because the point of this module is that adding one
is a dependency install plus a registry entry, not a refactor.

Gemini is a remote API: retrieved repository excerpts are sent to Google when
a question is answered. The key is read from ``GEMINI_API_KEY`` (or
``GOOGLE_API_KEY``), typically via ``backend/.env``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Pick up GEMINI_API_KEY and LLM_* from backend/.env when present. Real
# environment variables win over the file.
load_dotenv()

DEFAULT_PROVIDER = "gemini"
DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_TEMPERATURE = 0.1

#: Environment variables checked, in order, for the Gemini key.
GEMINI_KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


@dataclass(frozen=True)
class LLMSettings:
    """Resolved provider configuration."""

    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    api_key: str | None = None

    @property
    def is_local(self) -> bool:
        return False

    def describe(self) -> dict:
        """Safe for logs and API responses — never includes credentials."""
        return {
            "provider": self.provider,
            "model": self.model,
            "local": self.is_local,
            "temperature": self.temperature,
            "api_key_set": bool(self.api_key),
        }


def _api_key_from_env() -> str | None:
    for name in GEMINI_KEY_VARS:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return None


def settings_from_env(**overrides: Any) -> LLMSettings:
    """Read provider settings from the environment, with explicit overrides."""
    base = LLMSettings(
        provider=os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower(),
        model=os.getenv("LLM_MODEL", DEFAULT_MODEL).strip(),
        temperature=float(os.getenv("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)),
        api_key=_api_key_from_env(),
    )
    if not overrides:
        return base
    return LLMSettings(**{**base.__dict__, **{k: v for k, v in overrides.items() if v is not None}})


def _build_gemini(settings: LLMSettings):
    if not settings.api_key:
        raise RuntimeError(
            "LLM_PROVIDER='gemini' needs an API key. Set GEMINI_API_KEY in the "
            "environment or in backend/.env (get one at https://aistudio.google.com/apikey)."
        )
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise RuntimeError(
            "LLM_PROVIDER='gemini' needs the 'langchain-google-genai' package. "
            "Install it with: pip install langchain-google-genai"
        ) from exc

    return ChatGoogleGenerativeAI(
        model=settings.model,
        google_api_key=settings.api_key,
        temperature=settings.temperature,
    )


def _missing(package: str, provider: str) -> Callable[[LLMSettings], Any]:
    def build(settings: LLMSettings):
        raise RuntimeError(
            f"LLM_PROVIDER={provider!r} needs the {package!r} package. "
            f"Install it and set the provider's API key, or use LLM_PROVIDER=gemini."
        )

    return build


#: Provider name -> factory. Adding a provider is one entry plus its package.
PROVIDERS: dict[str, Callable[[LLMSettings], Any]] = {
    "gemini": _build_gemini,
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
