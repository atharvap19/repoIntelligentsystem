"""Repository retrieval.

Three components, composed rather than duplicated:

``VectorRetriever``
    Dense similarity over Chroma. Good at paraphrase, blind to exact names.

``KeywordRetriever``
    BM25 over a code-aware tokenisation of path, symbol and body. Good at
    exact names, blind to paraphrase.

``HybridRetriever``
    Fuses both candidate lists, then hands the pool to a reranker.

The split exists because measurement demanded it. Against the dense-only
index, "How does FastAPI handle routing?" returned zero chunks from the
library in its top 50 — 37 came from ``docs_src`` and 13 from ``tests``,
and 49 of the 50 were short ``module`` chunks. Dense similarity had no way
to know that the file literally named ``routing.py`` was the subject.

Nothing above this layer ever sees Chroma's parallel-array response shape.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, Sequence

from app.rag.bm25 import BM25Index
from app.rag.filters import build_where, matches, normalise_filters
from app.rag.vector_store import METADATA_FIELDS, VectorStore

logger = logging.getLogger(__name__)

#: Keys every retrieved chunk carries, whatever the retrieval strategy was.
RESULT_FIELDS: tuple[str, ...] = (
    "content",
    "distance",
    "score",
    "vector_score",
    "keyword_score",
    *METADATA_FIELDS,
)

#: How many candidates hybrid retrieval gathers per strategy before reranking.
DEFAULT_CANDIDATE_POOL = 25


def relevance_score(distance: float | None) -> float:
    """Map an unbounded distance onto ``(0, 1]``.

    The collection uses Chroma's default L2 space, so ``distance`` has no
    fixed upper bound and means little on its own. This is a monotonic
    rescaling for display and for fusion — not a probability.
    """
    if distance is None or distance < 0:
        return 0.0
    return 1.0 / (1.0 + float(distance))


def _blank_result(**overrides: Any) -> dict:
    result = {field: "" for field in METADATA_FIELDS}
    result.update({"content": "", "distance": 0.0, "score": 0.0,
                   "vector_score": 0.0, "keyword_score": 0.0})
    result.update(overrides)
    return result


class Reranker(Protocol):
    """Anything that can reorder a candidate pool.

    Deliberately minimal so a cross-encoder can replace the heuristic
    implementation without touching the retrievers.
    """

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]: ...


# --------------------------------------------------------------------------
# Dense
# --------------------------------------------------------------------------


class VectorRetriever:
    """Dense similarity retrieval over a repository's collection."""

    def __init__(self, vector_store: VectorStore | None = None):
        self.vector_store = vector_store or VectorStore()

    def retrieve(
        self,
        query_embedding: Sequence[float],
        repository: str,
        top_k: int = 5,
        filters: dict | None = None,
        where: dict | None = None,
    ) -> list[dict]:
        """Return the ``top_k`` nearest chunks in one repository, best first.

        Args:
            repository: which repository to search. Cross-repository bleed is
                prevented structurally — another collection is never opened.
            filters: optional metadata filters (see ``app.rag.filters``).
            where: raw Chroma clause, merged with ``filters``.
        """
        response = self.vector_store.search(
            repository=repository,
            embedding=query_embedding,
            top_k=top_k,
            where=build_where(filters, where),
        )
        return self._shape(response)

    @staticmethod
    def _shape(response: dict) -> list[dict]:
        """Flatten Chroma's parallel arrays into one dict per chunk.

        Chroma nests each field one level deep (one entry per query) and
        returns ``None`` rather than an empty list for an empty collection.
        """
        documents = (response.get("documents") or [[]])[0] or []
        metadatas = (response.get("metadatas") or [[]])[0] or []
        distances = (response.get("distances") or [[]])[0] or []

        results: list[dict] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}
            score = relevance_score(distance)
            results.append(
                _blank_result(
                    **{field: metadata.get(field, "") for field in METADATA_FIELDS},
                    content=document or "",
                    distance=float(distance),
                    score=score,
                    vector_score=score,
                )
            )
        return results


# --------------------------------------------------------------------------
# Sparse
# --------------------------------------------------------------------------


class KeywordRetriever:
    """BM25 retrieval over an in-memory mirror of a repository's chunks.

    The corpus is pulled out of Chroma once per repository and cached. At
    FastAPI's scale (7,235 chunks) the index builds in well under a second and
    costs a few tens of MB, which is a fair price for exact-name matching that
    dense vectors cannot provide.
    """

    def __init__(self, vector_store: VectorStore | None = None):
        self.vector_store = vector_store or VectorStore()
        self._indexes: dict[str, BM25Index] = {}

    def index_for(self, repository: str) -> BM25Index:
        """Build (or reuse) the BM25 index for a repository."""
        cached = self._indexes.get(repository)
        if cached is not None:
            return cached

        chunks = self.vector_store.all_chunks(repository)
        index = BM25Index.build(chunks)
        self._indexes[repository] = index
        logger.info("Built BM25 index for %r over %d chunks", repository, len(index))
        return index

    def invalidate(self, repository: str | None = None) -> None:
        """Drop cached indexes after the underlying collection changes."""
        if repository is None:
            self._indexes.clear()
        else:
            self._indexes.pop(repository, None)

    def retrieve(
        self,
        query: str,
        repository: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        """Return the ``top_k`` best keyword matches, best first."""
        index = self.index_for(repository)
        if not len(index):
            return []

        scored = index.search(query, top_k=max(top_k * 4, top_k))

        results: list[dict] = []
        for chunk, raw_score in scored:
            if filters and not matches(chunk, filters):
                continue
            results.append(
                _blank_result(
                    **{field: chunk.get(field, "") for field in METADATA_FIELDS},
                    content=chunk.get("content", ""),
                    distance=0.0,
                    score=raw_score,
                    keyword_score=raw_score,
                )
            )
            if len(results) >= top_k:
                break
        return results


# --------------------------------------------------------------------------
# Hybrid
# --------------------------------------------------------------------------


def _key(result: dict) -> tuple:
    return (result.get("relative_path", ""), result.get("chunk_index", 0))


def normalise(values: list[float]) -> list[float]:
    """Min-max a score list onto ``[0, 1]``.

    Dense scores and BM25 scores live on incomparable scales, so both are
    rescaled within the candidate pool before they are combined.
    """
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [1.0 if high > 0 else 0.0] * len(values)
    return [(v - low) / (high - low) for v in values]


class HybridRetriever:
    """Dense + BM25 candidate generation, followed by reranking."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        vector_retriever: VectorRetriever | None = None,
        keyword_retriever: KeywordRetriever | None = None,
        reranker: Reranker | None = None,
        candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    ):
        store = vector_store or VectorStore()
        self.vector_store = store
        self.vector = vector_retriever or VectorRetriever(store)
        self.keyword = keyword_retriever or KeywordRetriever(store)
        self.candidate_pool = candidate_pool

        if reranker is None:
            from app.rag.reranker import HeuristicReranker

            reranker = HeuristicReranker()
        self.reranker = reranker

    def invalidate(self, repository: str | None = None) -> None:
        self.keyword.invalidate(repository)

    def retrieve(
        self,
        query: str,
        query_embedding: Sequence[float],
        repository: str,
        top_k: int = 5,
        language: str | None = None,
        extension: str | None = None,
        chunk_type: str | None = None,
        filters: dict | None = None,
        where: dict | None = None,
    ) -> list[dict]:
        """Retrieve, fuse and rerank.

        Two-stage by design: gather ``candidate_pool`` candidates from each
        strategy, then let the reranker pick ``top_k``. Retrieving 5 directly
        gives the reranker nothing to work with.
        """
        active = dict(
            normalise_filters(language=language, extension=extension, chunk_type=chunk_type)
        )
        if filters:
            active.update(filters)

        pool = max(self.candidate_pool, top_k)

        dense = self.vector.retrieve(
            query_embedding=query_embedding,
            repository=repository,
            top_k=pool,
            filters=active or None,
            where=where,
        )
        sparse = self.keyword.retrieve(
            query=query,
            repository=repository,
            top_k=pool,
            filters=active or None,
        )

        candidates = self._fuse(dense, sparse)
        return self.reranker.rerank(query, candidates, top_k)

    @staticmethod
    def _fuse(dense: list[dict], sparse: list[dict]) -> list[dict]:
        """Union both lists, keeping each chunk's score from either strategy."""
        dense_scores = normalise([r["vector_score"] for r in dense])
        sparse_scores = normalise([r["keyword_score"] for r in sparse])

        merged: dict[tuple, dict] = {}

        for result, score in zip(dense, dense_scores):
            entry = dict(result)
            entry["vector_score"] = score
            entry["keyword_score"] = 0.0
            merged[_key(result)] = entry

        for result, score in zip(sparse, sparse_scores):
            key = _key(result)
            if key in merged:
                merged[key]["keyword_score"] = score
            else:
                entry = dict(result)
                entry["keyword_score"] = score
                entry["vector_score"] = 0.0
                # A keyword-only hit has no distance; leave it at 0.0 rather
                # than inventing one, and let the reranker weigh the scores.
                merged[key] = entry

        return list(merged.values())


# --------------------------------------------------------------------------
# Backwards-compatible facade
# --------------------------------------------------------------------------


class Retriever:
    """Dense-only retrieval, preserved for callers that pass no query text."""

    def __init__(self, vector_store: VectorStore | None = None):
        self.vector_store = vector_store or VectorStore()
        self._vector = VectorRetriever(self.vector_store)

    def retrieve(
        self,
        query_embedding: Sequence[float],
        repository: str,
        top_k: int = 5,
        language: str | None = None,
        extension: str | None = None,
        chunk_type: str | None = None,
        where: dict | None = None,
    ) -> list[dict]:
        filters = normalise_filters(
            language=language, extension=extension, chunk_type=chunk_type
        )
        return self._vector.retrieve(
            query_embedding=query_embedding,
            repository=repository,
            top_k=top_k,
            filters=filters or None,
            where=where,
        )
