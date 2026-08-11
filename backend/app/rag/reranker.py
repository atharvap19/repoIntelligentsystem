"""Second-stage reranking.

Hardware constraint drives the design: this runs beside a 4GB RTX 3050 that is
already hosting qwen3 and nomic-embed-text. A cross-encoder would contend for
that VRAM on every query, so the default reranker is a feature-based scorer —
no model, no VRAM, sub-millisecond.

It is deliberately a plug: ``HeuristicReranker`` satisfies the ``Reranker``
protocol in ``app.rag.retriever``, and a cross-encoder can be dropped in later
without touching retrieval.

The features encode what measurement showed dense similarity gets wrong on
this corpus:

* short ``module`` chunks from tutorials dominated the top 50 of a broad query
  (49 of 50), because short generic text scores well against short generic
  questions;
* the library itself is 4.8% of FastAPI's indexed files, so implementation is
  structurally outnumbered by tests and examples;
* the file that *is* the subject of a question is often named after it, a
  signal dense vectors discard entirely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.rag.tokenize import tokenize, tokenize_path

#: Prior for what a chunk *is*. A function or method is a self-contained
#: answer; a `module` chunk is usually imports and constants.
CHUNK_TYPE_PRIOR: dict[str, float] = {
    "method": 1.00,
    "function": 1.00,
    "class": 0.95,
    "block": 0.70,
    "module": 0.55,
}

#: Prior for where a chunk lives. Implementation should outrank the tests and
#: tutorials that merely exercise it — the explicit goal in the brief.
PATH_PRIORS: tuple[tuple[re.Pattern, float], ...] = (
    (re.compile(r"(^|/)(tests?|testing)(/|$)"), 0.55),
    (re.compile(r"(^|/)(docs?|docs_src|examples?|samples?|tutorial)"), 0.55),
    (re.compile(r"(^|/)(scripts?|tools?|benchmarks?)(/|$)"), 0.80),
    (re.compile(r"^test_|_test\.py$|_test$"), 0.60),
)

DEFAULT_PATH_PRIOR = 1.0


@dataclass
class RerankWeights:
    """Linear weights over the feature set.

    Chosen to be legible rather than tuned: dense and keyword evidence lead,
    symbol and path overlap break ties, and the priors demote whole categories.
    Item 9's benchmark is where these should be revisited against measurements
    rather than intuition.
    """

    vector: float = 1.00
    keyword: float = 0.90
    symbol_overlap: float = 0.60
    path_overlap: float = 0.45
    chunk_type_prior: float = 0.35
    path_prior: float = 0.50


@dataclass
class HeuristicReranker:
    """Feature-based reranker. No model, no VRAM."""

    weights: RerankWeights = field(default_factory=RerankWeights)

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Score, sort and truncate a candidate pool.

        Each candidate gains a ``rerank_score`` and a ``rerank_features``
        breakdown so a ranking decision can always be explained.
        """
        if not candidates:
            return []

        query_terms = set(tokenize(query))

        for candidate in candidates:
            features = self.features(query_terms, candidate)
            candidate["rerank_features"] = features
            candidate["rerank_score"] = self.combine(features)
            # `score` stays the surfaced ordering key for API consumers.
            candidate["score"] = candidate["rerank_score"]

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[:top_k]

    # -- features ----------------------------------------------------------

    def features(self, query_terms: set[str], candidate: dict) -> dict[str, float]:
        symbol_terms = set(tokenize(candidate.get("symbol", "")))
        path_terms = set(tokenize_path(candidate.get("relative_path", "")))

        return {
            "vector": float(candidate.get("vector_score", 0.0)),
            "keyword": float(candidate.get("keyword_score", 0.0)),
            "symbol_overlap": _overlap(query_terms, symbol_terms),
            "path_overlap": _overlap(query_terms, path_terms),
            "chunk_type_prior": CHUNK_TYPE_PRIOR.get(
                str(candidate.get("chunk_type", "")), 0.75
            ),
            "path_prior": path_prior(str(candidate.get("relative_path", ""))),
        }

    def combine(self, features: dict[str, float]) -> float:
        weights = self.weights
        return (
            weights.vector * features["vector"]
            + weights.keyword * features["keyword"]
            + weights.symbol_overlap * features["symbol_overlap"]
            + weights.path_overlap * features["path_overlap"]
            + weights.chunk_type_prior * features["chunk_type_prior"]
            + weights.path_prior * features["path_prior"]
        )


def _overlap(query_terms: set[str], candidate_terms: set[str]) -> float:
    """Fraction of the query's terms present in the candidate's terms."""
    if not query_terms or not candidate_terms:
        return 0.0
    return len(query_terms & candidate_terms) / len(query_terms)


def path_prior(relative_path: str) -> float:
    """Lowest matching prior for a path; 1.0 for ordinary source."""
    if not relative_path:
        return DEFAULT_PATH_PRIOR
    lowered = relative_path.lower()
    prior = DEFAULT_PATH_PRIOR
    for pattern, value in PATH_PRIORS:
        if pattern.search(lowered):
            prior = min(prior, value)
    return prior


class FusionOnlyReranker:
    """Ranks by summed dense + keyword score, with none of the heuristics.

    The benchmark control that isolates what the feature weights actually buy.
    Sorting by ``vector_score`` alone would silently discard keyword-only hits
    — they carry no distance — which would make the control a dense-only run
    wearing a hybrid label.
    """

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        for candidate in candidates:
            fused = float(candidate.get("vector_score", 0.0)) + float(
                candidate.get("keyword_score", 0.0)
            )
            candidate["rerank_score"] = fused
            candidate["score"] = fused
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[:top_k]


#: Retained name for the control used before the fusion fix.
IdentityReranker = FusionOnlyReranker
