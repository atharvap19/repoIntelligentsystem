"""Agent endpoint — the chat that talks to the knowledge graph.

Runs the full pipeline — hybrid retrieval, graph neighbourhood, the user's
graph selection and conversation state — and returns, beside the answer, the
graph nodes that answer drew on (``highlight``). The frontend lights those up
in the knowledge graph, so every answer visibly changes the graph.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent.graph import RepositoryAgent
from app.agent.intents import INTENTS
from app.agent.navigation import TARGET_TYPES
from app.agent.response import (
    UnknownRepositoryError,
    as_source,
    graph_highlight,
    resolve_repository,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])

agent = RepositoryAgent()


class ExplorerContextPayload(BaseModel):
    """What the user currently has selected in the graph.

    Every field is optional. ``node_id`` is the precise field — the rest are
    labels the model reads.
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
    #: depend on?" has no prior turn to resolve against.
    conversation_id: str = Field(default="default", min_length=1, max_length=120)
    explorer_context: ExplorerContextPayload | None = None


def _available_repositories() -> list[str]:
    names = set(agent.graph_service.store.repositories())
    names.update(agent.search.retriever.vector_store.list_repositories())
    return sorted(names)


@router.get("/intents")
def list_intents():
    """Intents the router can dispatch to, and navigation target types."""
    return {"intents": list(INTENTS), "target_types": list(TARGET_TYPES)}


@router.post("/ask")
def ask(request: AgentRequest):
    """Answer a question about a repository from its graph and code."""
    try:
        repository = resolve_repository(request.repository, _available_repositories())
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
        "navigation": result.get("navigation"),
        "ui_action": result.get("ui_action"),
        #: What to light up in the knowledge graph for this answer.
        "highlight": graph_highlight(result, repository),
        "sources": [as_source(chunk) for chunk in (result.get("sources") or [])],
        "context_stats": result.get("context_stats", {}),
        "trace": result.get("trace", []),
    }
