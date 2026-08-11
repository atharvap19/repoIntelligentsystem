"""Agent endpoint.

Sits alongside ``POST /chat/`` rather than replacing it: the Phase 2 route is
pure RAG and stays the fast path, while this one routes through LangGraph and
can answer structural, dependency and history questions that retrieval alone
cannot.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent.graph import RepositoryAgent
from app.agent.intents import INTENTS
from app.services.chat_service import ChatService, UnknownRepositoryError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])

agent = RepositoryAgent()
#: Reused purely for its repository-resolution rules, so both endpoints
#: behave identically when `repository` is omitted.
_resolver = ChatService(retriever=agent.search.retriever, llm_service=agent.llm)


class AgentRequest(BaseModel):
    question: str = Field(min_length=1)
    repository: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    #: Conversation key. Follow-up questions must reuse it, or "what does it
    #: depend on?" has no prior turn to resolve against.
    conversation_id: str = Field(default="default", min_length=1, max_length=120)


@router.get("/intents")
def list_intents():
    """Intents the router can dispatch to."""
    return {"intents": list(INTENTS)}


@router.post("/ask")
def ask(request: AgentRequest):
    """Answer a question through the LangGraph router."""
    try:
        repository = _resolver.resolve_repository(request.repository)
    except UnknownRepositoryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    result = agent.ask(
        question=request.question,
        repository=repository,
        conversation_id=request.conversation_id,
        top_k=request.top_k,
    )

    return {
        "question": request.question,
        "repository": repository,
        "conversation_id": request.conversation_id,
        "intent": result.get("intent"),
        "confidence": result.get("confidence"),
        "answer": result.get("answer", ""),
        "focus": {
            "node_id": result.get("focus_node_id"),
            "label": result.get("focus_label"),
        },
        # Drives graph highlighting / file opening in the UI.
        "ui_action": result.get("ui_action"),
        "graph": result.get("graph_payload"),
        "sources": [ChatService._as_source(chunk) for chunk in (result.get("sources") or [])],
        "trace": result.get("trace", []),
    }
