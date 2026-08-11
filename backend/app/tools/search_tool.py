"""Retrieval tool.

A thin adapter over the Phase 2 hybrid retriever. It exists so the agent has
one calling convention for every tool, not because retrieval needed wrapping —
all the ranking work stays in ``app.rag.retriever``.
"""

from __future__ import annotations

import logging
from typing import Any

from app.rag.retriever import HybridRetriever
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class SearchTool:
    """Semantic + keyword search over an indexed repository."""

    name = "search_repository"

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        embedding_service: EmbeddingService | None = None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.embedding_service = embedding_service or EmbeddingService()

    def run(
        self,
        query: str,
        repository: str,
        top_k: int = 5,
        language: str | None = None,
        chunk_type: str | None = None,
    ) -> list[dict]:
        """Retrieve the chunks most relevant to ``query``."""
        embedding = self.embedding_service.generate_embedding(query)
        results = self.retriever.retrieve(
            query=query,
            query_embedding=embedding,
            repository=repository,
            top_k=top_k,
            language=language,
            chunk_type=chunk_type,
        )
        logger.info("search_repository(%r) -> %d chunks", query[:60], len(results))
        return results

    def invalidate(self, repository: str | None = None) -> None:
        self.retriever.invalidate(repository)
