"""LLM service.

Answer generation runs as a LangChain LCEL pipeline::

    prompt | ChatOllama | StrOutputParser

That is the part of the stack where LangChain earns its place: prompt
templating, message construction, streaming and output parsing are generic
infrastructure with no repository-specific logic in them. Retrieval,
chunking and repository management stay custom — see ``docs/langchain.md``
for the per-component evaluation.

The chain deliberately stops at "text in, text out". Orchestration —
resolving which repository to search, retrieving, shaping the response —
belongs to ``ChatService``, not to a single giant chain.
"""

from __future__ import annotations

import logging
import re
from typing import Iterator, Sequence

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.prompts.system_prompt import SYSTEM_PROMPT, USER_PROMPT
from app.services.llm_provider import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    build_chat_model,
    settings_from_env,
)

logger = logging.getLogger(__name__)

#: qwen3 emits its reasoning inside <think> tags before the answer.
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)

#: How a chunk's location is described to the model.
_ROLE_BY_PATH = (
    (re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*$"), "test"),
    (re.compile(r"(^|/)(docs_src|examples?|samples?|tutorial)"), "example"),
    (re.compile(r"(^|/)docs?(/|$)"), "documentation"),
)


def describe_role(relative_path: str, chunk_type: str = "") -> str:
    """Label an excerpt so the prompt's precedence rules have something to act on."""
    lowered = (relative_path or "").lower()
    for pattern, role in _ROLE_BY_PATH:
        if pattern.search(lowered):
            return role
    return "implementation"


def strip_reasoning(text: str) -> str:
    """Remove qwen3's ``<think>`` block from a completion.

    Without this the reasoning trace is shown to the user as if it were the
    answer, and it is frequently longer than the answer itself.
    """
    cleaned = _THINK_BLOCK.sub("", text or "")
    # A truncated completion can open the block without closing it.
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned.strip()


class LLMService:
    """Generates answers from retrieved context via a local Ollama model."""

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        temperature: float = 0.1,
        llm=None,
    ):
        # Provider, model and host come from configuration (LLM_PROVIDER,
        # LLM_MODEL, LLM_HOST) with Ollama as the default, so nothing in the
        # application binds to a specific runtime.
        self.settings = settings_from_env(model=model, host=host, temperature=temperature)
        self.model = self.settings.model
        self.host = self.settings.host

        # `llm` is injectable so tests can exercise the chain without Ollama.
        self.llm = llm or build_chat_model(self.settings)

        self.prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    # -- context -----------------------------------------------------------

    @staticmethod
    def build_context(retrieved_chunks: Sequence[dict]) -> str:
        """Render retrieved chunks with their provenance.

        Each excerpt is numbered and labelled with path, symbol and role, so
        the model can apply the precedence rules in the system prompt instead
        of guessing which snippet is authoritative.
        """
        blocks: list[str] = []
        for position, chunk in enumerate(retrieved_chunks, start=1):
            path = chunk.get("relative_path", "unknown")
            symbol = chunk.get("symbol") or ""
            chunk_type = chunk.get("chunk_type") or ""
            start = chunk.get("start_line")
            end = chunk.get("end_line")

            header = f"[{position}] {path}"
            if start and end:
                header += f":{start}-{end}"
            if symbol:
                header += f"  symbol: {symbol}"
            if chunk_type:
                header += f"  kind: {chunk_type}"
            header += f"  role: {describe_role(path, chunk_type)}"

            blocks.append(f"{header}\n```\n{chunk.get('content', '')}\n```")

        return "\n\n".join(blocks) if blocks else "(no excerpts retrieved)"

    # -- generation --------------------------------------------------------

    def generate_response(
        self,
        question: str,
        retrieved_chunks: Sequence[dict],
        repository: str = "",
    ) -> str:
        """Run the LCEL chain and return the cleaned answer."""
        if not retrieved_chunks:
            return "I couldn't find that in the indexed repository."

        raw = self.chain.invoke(
            {
                "question": question,
                "repository": repository or "(unspecified)",
                "context": self.build_context(retrieved_chunks),
            }
        )
        return strip_reasoning(raw)

    def stream_response(
        self,
        question: str,
        retrieved_chunks: Sequence[dict],
        repository: str = "",
    ) -> Iterator[str]:
        """Token stream for the same chain.

        Reasoning cannot be stripped mid-stream, so callers that need clean
        output should use ``generate_response``.
        """
        yield from self.chain.stream(
            {
                "question": question,
                "repository": repository or "(unspecified)",
                "context": self.build_context(retrieved_chunks),
            }
        )
