"""Metadata filter construction for Chroma queries.

Kept separate from the retrievers so every retrieval strategy builds ``where``
clauses the same way, and so the Chroma-specific operator syntax lives in one
place.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

#: Metadata fields a caller may filter on. Repository is absent on purpose —
#: it is enforced by collection choice, not by a filter that could be omitted.
FILTERABLE_FIELDS: tuple[str, ...] = (
    "language",
    "extension",
    "chunk_type",
    "file_name",
    "relative_path",
    "symbol",
)


def _clause(field: str, value: Any) -> dict:
    """One equality or membership clause."""
    if isinstance(value, (list, tuple, set, frozenset)):
        values = [v for v in value if v is not None]
        if not values:
            raise ValueError(f"filter {field!r} was given an empty collection")
        if len(values) == 1:
            return {field: {"$eq": values[0]}}
        return {field: {"$in": sorted(values, key=str)}}
    return {field: {"$eq": value}}


def build_where(
    filters: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict | None:
    """Translate application filters into a Chroma ``where`` clause.

    ``None`` values are dropped so callers can pass optional arguments
    straight through. Returns ``None`` when nothing is being filtered, which
    Chroma treats as "no restriction".

    Raises:
        ValueError: on an unknown field or an empty value list.
    """
    clauses: list[dict] = []

    for field, value in (filters or {}).items():
        if value is None:
            continue
        if field not in FILTERABLE_FIELDS:
            raise ValueError(
                f"{field!r} is not filterable; expected one of {', '.join(FILTERABLE_FIELDS)}"
            )
        clauses.append(_clause(field, value))

    # `extra` is an escape hatch for raw Chroma syntax, merged unvalidated.
    if extra:
        clauses.append(dict(extra))

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def matches(metadata: Mapping[str, Any], filters: Mapping[str, Any] | None) -> bool:
    """Evaluate the same filters in Python.

    Needed by the keyword retriever, which ranks an in-memory corpus rather
    than issuing a Chroma query, so both halves of hybrid search agree on what
    a filter means.
    """
    for field, value in (filters or {}).items():
        if value is None:
            continue
        actual = metadata.get(field)
        if isinstance(value, (list, tuple, set, frozenset)):
            if actual not in set(value):
                return False
        elif actual != value:
            return False
    return True


def normalise_filters(**kwargs: Any) -> dict[str, Any]:
    """Drop ``None`` entries from keyword-style filter arguments."""
    return {k: v for k, v in kwargs.items() if v is not None}


def iter_filterable(values: Iterable[str]) -> list[str]:
    return [v for v in values if v in FILTERABLE_FIELDS]
