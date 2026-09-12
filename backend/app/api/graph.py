"""Repository graph endpoints.

Node IDs contain both ``::`` and ``/`` (``fastapi::file::fastapi/routing.py``),
so they travel as query parameters rather than path segments — a path
parameter would need ``:path`` plus encoding on every call for no benefit.

Reads are one level of the hierarchy, with one exception: ``/knowledge``
serves the whole repository for the knowledge graph, capped by node count.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from app.services.graph_service import (
    DEFAULT_CHILD_LIMIT,
    DEFAULT_KNOWLEDGE_LIMIT,
    GraphNotFoundError,
    GraphService,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["Graph"])

service = GraphService()


def _guard(call):
    """Map a missing repository or node onto a 404."""
    try:
        return call()
    except GraphNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/repositories")
def list_repositories():
    """Repositories that have a built graph, with their node counts."""
    return {"repositories": service.repositories()}


@router.get("/{repository}/overview")
def overview(repository: str):
    """Top-level view: the repository, its modules, and module dependencies."""
    return _guard(lambda: service.overview(repository))


@router.get("/{repository}/knowledge")
def knowledge(
    repository: str,
    limit: int = Query(DEFAULT_KNOWLEDGE_LIMIT, ge=50, le=4000),
):
    """Files, symbols and their relationships across the whole repository."""
    return _guard(lambda: service.knowledge_graph(repository, limit=limit))


@router.get("/{repository}/timeline")
def timeline(repository: str):
    """Module activity over time, releases and branches."""
    return _guard(lambda: service.timeline(repository))


@router.get("/{repository}/commits")
def commits(
    repository: str,
    limit: int = Query(60, ge=1, le=300),
):
    """Commits as timeline dots, oldest first, with file counts and releases."""
    return _guard(lambda: service.commit_timeline(repository, limit=limit))


@router.get("/{repository}/commit")
def commit_detail(repository: str, sha: str = Query(..., min_length=4)):
    """One commit: author, date, message, and the files it changed."""
    return _guard(lambda: service.commit_detail(repository, sha))


@router.get("/{repository}/commit/snapshot")
def commit_snapshot(repository: str, sha: str = Query(..., min_length=4)):
    """Approximate repository state as of a commit."""
    return _guard(lambda: service.snapshot_at_commit(repository, sha))


@router.get("/{repository}/changes")
def changes(
    repository: str,
    from_sha: str = Query(..., min_length=4),
    to_sha: str = Query(..., min_length=4),
):
    """What changed between two commits."""
    return _guard(lambda: service.changes_between(repository, from_sha, to_sha))


@router.get("/{repository}/hotspots")
def hotspots(repository: str, limit: int = Query(8, ge=1, le=30)):
    """Most depended-on, most changed and largest files, each with evidence."""
    return {"hotspots": _guard(lambda: service.hotspots(repository, limit=limit))}


@router.get("/{repository}/flow")
def flow(
    repository: str,
    node_id: str | None = Query(None, description="Start here instead of guessing"),
):
    """A static path through the import graph from an entry point."""
    return _guard(lambda: service.flow(repository, node_id))


@router.get("/nodes")
def node_set(id: list[str] = Query(..., description="Repeat once per node")):
    """A named set of nodes and the edges among them.

    For views that legitimately cross hierarchy levels — a derived flow spans
    several directories — where expanding one parent would show part of the
    answer and hide the rest.
    """
    return _guard(lambda: service.node_set(id))


@router.get("/node/path")
def node_path(node_id: str = Query(...)):
    """Ancestors from the repository root down to this node, root first."""
    return {"path": _guard(lambda: service.ancestors(node_id))}


@router.get("/node/neighborhood")
def neighborhood(
    node_id: str = Query(...),
    hops: int = Query(1, ge=1, le=3),
    limit: int = Query(40, ge=1, le=120),
):
    """The bounded subgraph around a node — what [Explore this] renders."""
    return _guard(lambda: service.neighborhood(node_id, hops=hops, limit=limit))


@router.get("/{repository}/snapshot")
def snapshot(
    repository: str,
    timestamp: int = Query(..., description="Unix seconds for the timeline cursor"),
):
    """Approximate repository state at a point in time."""
    return _guard(lambda: service.snapshot(repository, timestamp))


@router.get("/{repository}/search")
def search(
    repository: str,
    q: str = Query(..., min_length=1),
    limit: int = Query(25, ge=1, le=100),
):
    """Find nodes by name or path."""
    return {"results": service.search(repository, q, limit=limit)}


@router.get("/node/expand")
def expand(
    node_id: str = Query(..., description="Node to expand"),
    limit: int = Query(DEFAULT_CHILD_LIMIT, ge=1, le=500),
):
    """One level of children, with edges among them."""
    return _guard(lambda: service.expand(node_id, limit=limit))


@router.get("/node")
def node_detail(node_id: str = Query(...)):
    """File or symbol intelligence: symbols, dependencies, dependents, history."""
    return _guard(lambda: service.node_detail(node_id))


@router.get("/node/dependencies")
def dependencies(node_id: str = Query(...)):
    """Both directions at once, for graph highlighting."""
    return _guard(lambda: service.dependencies(node_id))


@router.get("/node/content")
def content(node_id: str = Query(...)):
    """File source for the code cell. Passes through secret redaction."""
    return _guard(lambda: service.file_content(node_id))


@router.get("/node/history")
def history(node_id: str = Query(...), limit: int = Query(20, ge=1, le=100)):
    """Commits that touched this file."""
    return {"history": _guard(lambda: service.history(node_id, limit=limit))}
