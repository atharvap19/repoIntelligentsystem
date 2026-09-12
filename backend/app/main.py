"""FastAPI application entrypoint."""

from fastapi import FastAPI

from app.api.agent import router as agent_router
from app.api.graph import router as graph_router
from app.api.indexing import router as index_router
from app.api.repositories import router as repositories_router


app = FastAPI(
    title="Repository Intelligence Platform",
    description="AI-powered Software Engineering Copilot",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "status": "running",
        "service": "Repository Intelligence Platform",
    }


app.include_router(agent_router)
app.include_router(graph_router)
app.include_router(index_router)
app.include_router(repositories_router)
