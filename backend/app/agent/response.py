"""Shaping an agent turn for the HTTP response.

Kept apart from ``app/api/agent.py`` because importing that module builds the
live agent against the real stores; these functions are pure and tested
directly.
"""

from __future__ import annotations

from app.graph.model import NODE_FILE, node_id, split_node_id
from app.services.graph_service import KNOWLEDGE_KINDS

#: Nodes one answer may highlight. The retrieved neighbourhood is already
#: bounded (24 nodes); sources and structural analyses add a few dozen at most.
MAX_HIGHLIGHT = 60


class UnknownRepositoryError(ValueError):
    """Raised when the question cannot be attached to an imported repository."""


def resolve_repository(requested: str | None, available: list[str]) -> str:
    """Pick the repository to answer about, or explain why it cannot be picked.

    ``available`` covers repositories with a graph *or* a code index: a
    freshly imported repository has its graph long before embedding finishes,
    and the agent can already answer from the graph alone.
    """
    if requested:
        if requested not in available:
            raise UnknownRepositoryError(
                f"Repository {requested!r} has not been imported. "
                f"Available: {', '.join(available) or 'none'}."
            )
        return requested
    if len(available) == 1:
        return available[0]
    if not available:
        raise UnknownRepositoryError("No repositories yet. Import one from a GitHub URL first.")
    raise UnknownRepositoryError(
        "Several repositories are imported, so 'repository' is required. "
        f"Available: {', '.join(available)}."
    )


def graph_highlight(result: dict, repository: str) -> dict:
    """The knowledge-graph nodes an answer drew on, most central first.

    Order matters because the list is capped: the resolved subject, then the
    graph seeds and whatever a structural analysis singled out (dependencies,
    a flow, hotspots), then the files the quoted code came from, and finally
    the rest of the retrieved neighbourhood. Kinds the knowledge graph does
    not draw — modules, directories, external packages — are skipped.
    """
    ordered: list[str] = []
    seen: set[str] = set()

    def add(identifier: str | None) -> None:
        if not identifier or identifier in seen:
            return
        try:
            _, kind, _ = split_node_id(identifier)
        except ValueError:
            return
        if kind in KNOWLEDGE_KINDS:
            seen.add(identifier)
            ordered.append(identifier)

    context = result.get("graph_context")
    action = result.get("ui_action") or {}
    navigation = result.get("navigation") or {}

    add(result.get("focus_node_id"))
    for identifier in getattr(context, "seeds", None) or []:
        add(identifier)
    add(action.get("node_id"))
    for key in ("dependencies", "dependents", "node_ids"):
        for identifier in action.get(key) or []:
            add(identifier)
    add(navigation.get("node_id"))
    for identifier in navigation.get("related_node_ids") or []:
        add(identifier)
    for identifier in result.get("navigation_nodes") or []:
        add(identifier)
    for chunk in result.get("sources") or []:
        path = chunk.get("relative_path")
        if path:
            add(node_id(repository, NODE_FILE, path))
    for hit in getattr(context, "hits", None) or []:
        add(hit.id)

    node_ids = ordered[:MAX_HIGHLIGHT]
    return {"node_ids": node_ids, "focus_node_id": node_ids[0] if node_ids else None}


def as_source(chunk: dict) -> dict:
    """Shape one retrieved chunk for the API response."""
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
