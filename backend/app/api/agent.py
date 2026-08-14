"""Agent endpoint.

Sits alongside ``POST /chat/`` rather than replacing it: the Phase 2 route is
pure RAG and stays available as the fast path, while this one runs the full
Phase 4 pipeline — hybrid retrieval, graph neighbourhood, Explorer context and
conversation state — and returns a structured navigation target beside the
answer.

The frontend uses this route for *both* the Chat view and the Explorer
sidebar. There is one conversation (Part 4), so there is one endpoint and one
``conversation_id`` behind it; which view the user happens to be looking at
changes only what is sent in ``explorer_context``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent.graph import RepositoryAgent
from app.agent.intents import INTENTS
from app.agent.navigation import TARGET_TYPES
from app.services.chat_service import ChatService, UnknownRepositoryError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])

agent = RepositoryAgent()
#: Reused purely for its repository-resolution rules, so both endpoints
#: behave identically when `repository` is omitted.
_resolver = ChatService(retriever=agent.search.retriever, llm_service=agent.llm)


class ExplorerContextPayload(BaseModel):
    """What the user currently has open (Part 3).

    Every field is optional. The Chat view sends an empty object; Explorer
    sends whatever is selected. ``node_id`` is the precise field — the rest
    are labels the model reads.
    """

    repository: str | None = None
    module: str | None = None
    file: str | None = None
    symbol: str | None = None
    node_id: str | None = None
    focus_node_ids: list[str] = Field(default_factory=list, max_length=40)
    commit_sha: str | None = None
    timestamp: int | None = None


class AgentRequest(BaseModel):
    question: str = Field(min_length=1)
    repository: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    #: Conversation key. Follow-up questions must reuse it, or "what does it
    #: depend on?" has no prior turn to resolve against. Chat and Explorer
    #: share one value so the conversation survives switching views.
    conversation_id: str = Field(default="default", min_length=1, max_length=120)
    explorer_context: ExplorerContextPayload | None = None


@router.get("/intents")
def list_intents():
    """Intents the router can dispatch to, and navigation target types."""
    return {"intents": list(INTENTS), "target_types": list(TARGET_TYPES)}


@router.post("/ask")
def ask(request: AgentRequest):
    """Answer a question through the Phase 4 pipeline."""
    try:
        repository = _resolver.resolve_repository(request.repository)
    except UnknownRepositoryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    explorer = (
        request.explorer_context.model_dump() if request.explorer_context else {}
    )

    result = agent.ask(
        question=request.question,
        repository=repository,
        conversation_id=request.conversation_id,
        top_k=request.top_k,
        explorer=explorer,
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
        #: Structured exploration target (Part 6). The frontend navigates from
        #: this, never by parsing the answer text.
        "navigation": result.get("navigation"),
        # Retained from Phase 3: applied automatically when the user is
        # already in Explorer. `navigation` is the offered, opt-in version.
        "ui_action": result.get("ui_action"),
        "graph": result.get("graph_payload"),
        "sources": [ChatService._as_source(chunk) for chunk in (result.get("sources") or [])],
        "context_stats": result.get("context_stats", {}),
        "trace": result.get("trace", []),
    }
