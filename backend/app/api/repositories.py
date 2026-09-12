"""Repository endpoints for the UI: import a GitHub URL, watch it become a graph."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.agent import agent
from app.api.indexing import service as indexing_service
from app.services.github_service import InvalidRepositoryUrl
from app.services.import_service import ImportService

router = APIRouter(prefix="/repositories", tags=["Repositories"])


def _invalidate_query_caches(repository: str) -> None:
    """Drop the agent's in-memory BM25 mirror once the corpus has changed."""
    invalidate = getattr(agent.search.retriever, "invalidate", None)
    if callable(invalidate):
        invalidate(repository)


service = ImportService(indexing=indexing_service, on_indexed=_invalidate_query_caches)


class ImportRequest(BaseModel):
    url: str


@router.get("")
def list_repositories():
    """Every repository with a graph or an import in progress."""
    graph_store = indexing_service.graph_store
    names = set(graph_store.repositories() if graph_store is not None else [])
    names.update(job.repository for job in service.jobs())

    repositories = []
    for name in sorted(names, key=str.lower):
        status = service.status(name) or {}
        repositories.append(
            {
                "repository": name,
                "counts": graph_store.counts(name) if status.get("graph_ready") else {},
                "status": status,
            }
        )
    return {"repositories": repositories}


@router.post("/import")
def import_repository(request: ImportRequest):
    """Clone, build the graph, index code. Returns at once; poll the status."""
    try:
        job = service.start(request.url)
    except InvalidRepositoryUrl as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return job.to_dict()


@router.get("/{repository}/status")
def repository_status(repository: str):
    status = service.status(repository)
    if status is None:
        raise HTTPException(status_code=404, detail=f"{repository!r} has not been imported.")
    return status
