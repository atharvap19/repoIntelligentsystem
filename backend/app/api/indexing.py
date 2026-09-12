"""Indexing endpoints.

Indexing is synchronous and can run for minutes on a large repository — the
FastAPI checkout takes ~5 minutes to embed. That is acceptable for a local
single-user tool and keeps the control flow honest; a job queue belongs here
before this is ever exposed to more than one user.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.agent import agent
from app.services.indexing_service import IndexingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/index", tags=["Index"])

service = IndexingService(batch_size=64)


class IndexRequest(BaseModel):
    #: Path to a checkout. Optional when `repository` names one already under
    #: the repositories root.
    path: str | None = None
    repository: str | None = None


def _resolve(request: IndexRequest) -> tuple[Path, str]:
    """Work out which checkout to index and what to call it."""
    if request.path:
        path = Path(request.path)
    elif request.repository:
        path = service.repositories_root / request.repository
    else:
        raise HTTPException(status_code=422, detail="Provide 'path' or 'repository'.")

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Checkout not found: {path}")
    if not path.is_dir():
        raise HTTPException(status_code=422, detail=f"Not a directory: {path}")

    return path, request.repository or path.name


def _refresh_query_caches(repository: str) -> None:
    """Drop the query path's BM25 mirror after the corpus changes.

    The agent holds its own retriever with an in-memory index; without this it
    would keep serving chunks that indexing has just replaced.
    """
    invalidate = getattr(agent.search.retriever, "invalidate", None)
    if callable(invalidate):
        invalidate(repository)


@router.post("/")
def index_repository(request: IndexRequest):
    """Build a repository's index from scratch."""
    path, name = _resolve(request)
    report = service.index_repository(path, repository=name)
    _refresh_query_caches(name)
    return report


@router.post("/update")
def update_repository(request: IndexRequest):
    """Re-index only the files that changed since the last run."""
    path, name = _resolve(request)
    report = service.update_repository(path, repository=name)
    _refresh_query_caches(name)
    return report


@router.post("/reindex")
def reindex_repository(request: IndexRequest):
    """Drop the collection and rebuild it."""
    path, name = _resolve(request)
    report = service.reindex_repository(name, repository_path=path)
    _refresh_query_caches(name)
    return report


@router.delete("/{repository}")
def delete_repository(repository: str):
    """Remove a repository's collection."""
    result = service.delete_repository(repository)
    _refresh_query_caches(repository)
    if not result["deleted"]:
        raise HTTPException(status_code=404, detail=f"{repository!r} is not indexed.")
    return result
