"""Live graph channel — lets an outside agent drive the knowledge graph.

The MCP server (``mcp_server.py``) posts to ``/live/show`` whenever Claude
looks at part of a repository; every open browser tab listens on
``/live/events`` (Server-Sent Events) and lights those nodes up. The channel
is in-memory and one-way: nothing is stored, and a tab opened later only sees
events from then on.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/live", tags=["Live"])

#: Seconds between keep-alive comments, so proxies do not close an idle stream.
KEEPALIVE_SECONDS = 15

_subscribers: set[asyncio.Queue] = set()


class ShowRequest(BaseModel):
    repository: str = Field(min_length=1)
    node_ids: list[str] = Field(default_factory=list, max_length=60)
    focus_node_id: str | None = None
    #: Short text shown in the chat panel beside the highlight.
    note: str = ""


@router.post("/show")
async def show(request: ShowRequest):
    """Broadcast a highlight to every connected graph view."""
    payload = request.model_dump()
    for queue in list(_subscribers):
        queue.put_nowait(payload)
    return {"delivered_to": len(_subscribers)}


@router.get("/events")
async def events(request: Request):
    """Server-Sent Events stream of highlights."""
    queue: asyncio.Queue = asyncio.Queue()
    _subscribers.add(queue)

    async def stream():
        try:
            yield ": connected\n\n"
            while not await request.is_disconnected():
                try:
                    payload = await asyncio.wait_for(queue.get(), KEEPALIVE_SECONDS)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps(payload)}\n\n"
        finally:
            _subscribers.discard(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
