from fastapi import APIRouter
from pydantic import BaseModel

from app.services.chat_service import ChatService

router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5


service = ChatService()


@router.post("/")
def chat(request: ChatRequest):
    """
    Ask a question about the indexed repository.
    """

    return service.chat(
        question=request.question,
        top_k=request.top_k
    )