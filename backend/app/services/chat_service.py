"""Chat service.

Application-level orchestration for the RAG pipeline. Deliberately *not* a
single LangChain chain: resolving which repository to search, running hybrid
retrieval, and shaping the API response are application concerns. The chain
starts where the prompt starts — see ``LLMService``.
"""

from __future__ import annotations

import logging
import time

from app.rag.retriever import HybridRetriever
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class UnknownRepositoryError(ValueError):
    """Raised when the question cannot be attached to an indexed repository."""


class ChatService:
    """Coordinates the complete RAG pipeline."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        retriever: HybridRetriever | None = None,
        llm_service: LLMService | None = None,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.retriever = retriever or HybridRetriever()
        self.llm_service = llm_service or LLMService()

    def chat(
        self,
        question: str,
        top_k: int = 5,
        repository: str | None = None,
        language: str | None = None,
        extension: str | None = None,
        chunk_type: str | None = None,
    ) -> dict:
        """Answer a question against one indexed repository.

        Args:
            repository: which repository to search. Optional: when exactly one
                repository is indexed it is used automatically, which keeps
                pre-Phase-2 clients working.
            language, extension, chunk_type: optional metadata filters.

        Raises:
            UnknownRepositoryError: if the repository is absent, or omitted
                while several are indexed.
        """
        target = self.resolve_repository(repository)
        timings: dict[str, float] = {}

        mark = time.perf_counter()
        query_embedding = self.embedding_service.generate_embedding(question)
        timings["embed"] = (time.perf_counter() - mark) * 1000

        mark = time.perf_counter()
        retrieved_chunks = self.retriever.retrieve(
            query=question,
            query_embedding=query_embedding,
            repository=target,
            top_k=top_k,
            language=language,
            extension=extension,
            chunk_type=chunk_type,
        )
        timings["search"] = (time.perf_counter() - mark) * 1000

        mark = time.perf_counter()
        answer = self.llm_service.generate_response(
            question=question,
            retrieved_chunks=retrieved_chunks,
            repository=target,
        )
        timings["generate"] = (time.perf_counter() - mark) * 1000

        return {
            "question": question,
            "repository": target,
            "answer": answer,
            "sources": [self._as_source(chunk) for chunk in retrieved_chunks],
            "timings": {k: round(v, 1) for k, v in timings.items()},
        }

    def resolve_repository(self, repository: str | None) -> str:
        """Pick the repository to search, or explain why it cannot be picked."""
        available = self.retriever.vector_store.list_repositories()

        if repository:
            if repository not in available:
                raise UnknownRepositoryError(
                    f"Repository {repository!r} is not indexed. "
                    f"Available: {', '.join(available) or 'none'}."
                )
            return repository

        if len(available) == 1:
            return available[0]
        if not available:
            raise UnknownRepositoryError(
                "No repositories are indexed yet. Import and index one first."
            )
        raise UnknownRepositoryError(
            "Several repositories are indexed, so 'repository' is required. "
            f"Available: {', '.join(available)}."
        )

    @staticmethod
    def _as_source(chunk: dict) -> dict:
        """Shape one retrieved chunk for the API response.

        ``file`` is kept as the primary path key so existing clients keep
        working; everything else is additive.
        """
        return {
            "file": chunk.get("relative_path", ""),
            "relative_path": chunk.get("relative_path", ""),
            "file_name": chunk.get("file_name", ""),
            "repository": chunk.get("repository", ""),
            "language": chunk.get("language", ""),
            "symbol": chunk.get("symbol", ""),
            "chunk_type": chunk.get("chunk_type", ""),
            "chunk_index": chunk.get("chunk_index", 0),
            "start_line": chunk.get("start_line", 0),
            "end_line": chunk.get("end_line", 0),
            "distance": chunk.get("distance", 0.0),
            "score": round(float(chunk.get("score", 0.0)), 4),
            "vector_score": round(float(chunk.get("vector_score", 0.0)), 4),
            "keyword_score": round(float(chunk.get("keyword_score", 0.0)), 4),
            "content": chunk.get("content", ""),
        }
