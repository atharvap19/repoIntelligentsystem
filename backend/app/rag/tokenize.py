"""Code-aware tokenisation for keyword search.

Natural-language tokenisers destroy the signal that matters in a repository.
``APIRouter.include_router`` has to yield ``api``, ``router``, ``include`` and
``includerouter``; ``fastapi/routing.py`` has to yield ``routing``. Without
that, a question about "routing" can never match the file literally named
after it — which is exactly the failure the dense retriever exhibits.
"""

from __future__ import annotations

import re

#: Split on non-alphanumerics, then again on camelCase and digit boundaries.
_SEPARATORS = re.compile(r"[^A-Za-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

#: Words that carry no discriminating power in a codebase question. Kept
#: deliberately small — over-pruning hurts more than it helps here.
STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "how", "what", "why", "when", "where", "which",
        "who", "this", "that", "these", "those", "it", "its", "to", "of", "in",
        "on", "at", "for", "with", "and", "or", "not", "as", "by", "from",
        "can", "could", "should", "would", "will", "i", "you", "we", "they",
        "me", "my", "your", "our", "there", "here", "about", "into", "over",
        "then", "than", "so", "if", "else", "return", "self",
        # Corpus meta-vocabulary. "this repository" refers to the corpus, but
        # BM25 cannot know that: on FastAPI, "repository" matched
        # `CommentsRepository` and `AllDiscussionsRepository` in maintainer
        # scripts, so "what is this repository about" retrieved translation
        # tooling. These words never discriminate between chunks.
        "repository", "repositories", "repo", "repos", "project", "codebase",
        "code", "codebases", "package", "library", "app", "application",
    }
)


def split_identifier(text: str) -> list[str]:
    """``includeRouter`` -> ``['include', 'router', 'includerouter']``."""
    tokens: list[str] = []
    for part in _SEPARATORS.split(text):
        if not part:
            continue
        lowered = part.lower()
        pieces = [p.lower() for p in _CAMEL.split(part) if p]
        tokens.extend(pieces)
        # Keep the joined form too, so an exact identifier still matches.
        if len(pieces) > 1:
            tokens.append(lowered)
        elif not pieces:
            tokens.append(lowered)
    return tokens


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    """Tokenise free text or source code into searchable terms."""
    if not text:
        return []
    tokens = split_identifier(text)
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    else:
        tokens = [t for t in tokens if len(t) > 1]
    return tokens


def tokenize_path(relative_path: str) -> list[str]:
    """Tokens from a path, minus the extension.

    ``fastapi/routing.py`` -> ``['fastapi', 'routing']``
    """
    without_extension = re.sub(r"\.[A-Za-z0-9]+$", "", relative_path or "")
    return tokenize(without_extension)


def chunk_terms(chunk: dict, path_weight: int = 3, symbol_weight: int = 4) -> list[str]:
    """Build the term list BM25 indexes for one chunk.

    Path and symbol tokens are repeated rather than scored separately: BM25
    already handles term frequency, so repetition is the natural way to say
    "a hit here counts for more than a hit in the body". The weights are
    deliberately modest — enough to lift the file that *is* the topic above
    files that merely mention it.
    """
    terms: list[str] = []
    terms.extend(tokenize_path(chunk.get("relative_path", "")) * path_weight)
    terms.extend(tokenize(chunk.get("symbol", "")) * symbol_weight)
    terms.extend(tokenize(chunk.get("content", ""), drop_stopwords=True))
    return terms
