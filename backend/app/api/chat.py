from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.chat_service import ChatService, UnknownRepositoryError

router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    #: Optional so pre-Phase-2 clients keep working: with a single indexed
    #: repository the service resolves it automatically.
    repository: str | None = None
    #: Optional metadata filters.
    language: str | None = None
    extension: str | None = None
    chunk_type: str | None = None


service = ChatService()


@router.post("/")
def chat(request: ChatRequest):
    """
    Ask a question about an indexed repository.
    """
    try:
        return service.chat(
            question=request.question,
            top_k=request.top_k,
            repository=request.repository,
            language=request.language,
            extension=request.extension,
            chunk_type=request.chunk_type,
        )
    except UnknownRepositoryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/repositories")
def list_repositories():
    """
    Repositories currently available to query.
    """
    return {"repositories": service.retriever.vector_store.list_repositories()}
