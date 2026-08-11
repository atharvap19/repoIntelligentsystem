"""Repository graph endpoints.

Node IDs contain both ``::`` and ``/`` (``fastapi::file::fastapi/routing.py``),
so they travel as query parameters rather than path segments — a path
parameter would need ``:path`` plus encoding on every call for no benefit.

Every read is one level of the hierarchy. There is deliberately no endpoint
that returns a whole repository graph.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from app.services.graph_service import (
    DEFAULT_CHILD_LIMIT,
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


@router.get("/{repository}/timeline")
def timeline(repository: str):
    """Module activity over time, releases and branches."""
    return _guard(lambda: service.timeline(repository))


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
