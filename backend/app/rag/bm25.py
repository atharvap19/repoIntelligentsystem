"""Okapi BM25 over an in-memory chunk corpus.

Implemented directly rather than pulled from a package: the ranking is 40
lines, and owning it means the tokenizer can be code-aware (see
``app.rag.tokenize``) instead of tuned for prose. No extra dependency, and no
model to load onto a 4GB GPU.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from app.rag.tokenize import chunk_terms, tokenize

# Standard Okapi parameters. k1 controls term-frequency saturation, b controls
# length normalisation.
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


@dataclass
class BM25Index:
    """A searchable BM25 index over chunk dictionaries."""

    k1: float = DEFAULT_K1
    b: float = DEFAULT_B

    chunks: list[dict] = field(default_factory=list)
    _term_frequencies: list[Counter] = field(default_factory=list)
    _lengths: list[int] = field(default_factory=list)
    _document_frequency: Counter = field(default_factory=Counter)
    _average_length: float = 0.0

    # -- construction ------------------------------------------------------

    @classmethod
    def build(cls, chunks: Iterable[dict], k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> "BM25Index":
        index = cls(k1=k1, b=b)
        for chunk in chunks:
            terms = chunk_terms(chunk)
            counts = Counter(terms)

            index.chunks.append(chunk)
            index._term_frequencies.append(counts)
            index._lengths.append(len(terms))
            for term in counts:
                index._document_frequency[term] += 1

        total = sum(index._lengths)
        index._average_length = total / len(index._lengths) if index._lengths else 0.0
        return index

    def __len__(self) -> int:
        return len(self.chunks)

    # -- scoring -----------------------------------------------------------

    def _idf(self, term: str) -> float:
        """Robertson-Sparck-Jones IDF with the usual +0.5 smoothing.

        The ``+1`` inside the log keeps the value positive for terms that
        appear in more than half the corpus, which the unsmoothed form makes
        negative — that would let a common term actively penalise a document.
        """
        n = len(self.chunks)
        df = self._document_frequency.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score_query(self, query: str) -> list[float]:
        """BM25 score of every chunk against ``query``."""
        return self.score_terms(tokenize(query))

    def score_terms(self, terms: Sequence[str]) -> list[float]:
        scores = [0.0] * len(self.chunks)
        if not terms or not self.chunks:
            return scores

        idf_cache = {term: self._idf(term) for term in set(terms)}

        for position, counts in enumerate(self._term_frequencies):
            length = self._lengths[position]
            if not length:
                continue
            norm = self.k1 * (
                1 - self.b + self.b * (length / self._average_length if self._average_length else 1.0)
            )
            total = 0.0
            for term in terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                total += idf_cache[term] * (frequency * (self.k1 + 1)) / (frequency + norm)
            scores[position] = total

        return scores

    def search(self, query: str, top_k: int = 20) -> list[tuple[dict, float]]:
        """Best ``top_k`` chunks for a query, highest score first."""
        scores = self.score_query(query)
        ranked = sorted(
            ((chunk, score) for chunk, score in zip(self.chunks, scores) if score > 0),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:top_k]
