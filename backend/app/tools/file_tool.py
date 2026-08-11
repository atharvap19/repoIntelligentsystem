"""File and symbol lookup tool.

Reads the repository graph rather than searching it. When a question names a
file or a symbol, guessing with embeddings is strictly worse than looking the
name up — this is that lookup.
"""

from __future__ import annotations

import logging

from app.graph.model import NODE_CLASS, NODE_FILE, NODE_FUNCTION, NODE_METHOD
from app.services.graph_service import GraphNotFoundError, GraphService

logger = logging.getLogger(__name__)


class FileTool:
    """Locate files and symbols, and read their content and metadata."""

    name = "inspect_file"

    def __init__(self, graph_service: GraphService | None = None):
        self.graph = graph_service or GraphService()

    def find(self, repository: str, term: str, limit: int = 8) -> list[dict]:
        """Nodes whose name or path matches ``term``.

        Exact filename matches are promoted: a search for ``routing.py`` should
        return that file before every symbol defined inside it.
        """
        results = self.graph.search(repository, term, limit=limit * 3)
        lowered = term.lower().strip()

        def rank(node: dict) -> tuple:
            name = node["name"].lower()
            return (
                0 if name == lowered else 1 if name.startswith(lowered) else 2,
                0 if node["kind"] == NODE_FILE else 1,
                len(name),
            )

        return sorted(results, key=rank)[:limit]

    def detail(self, node_id: str) -> dict:
        """Full intelligence for one node."""
        return self.graph.node_detail(node_id)

    def content(self, node_id: str) -> dict:
        """File source, redacted."""
        return self.graph.file_content(node_id)

    def symbols(self, repository: str, name: str) -> list[dict]:
        """Symbol nodes matching a name."""
        return [
            node
            for node in self.graph.search(repository, name, limit=30)
            if node["kind"] in (NODE_CLASS, NODE_FUNCTION, NODE_METHOD)
        ]

    def run(self, repository: str, term: str) -> dict:
        """Resolve ``term`` to a node and return everything known about it."""
        matches = self.find(repository, term)
        if not matches:
            return {"found": False, "term": term, "matches": []}

        best = matches[0]
        payload: dict = {"found": True, "term": term, "node": best, "matches": matches[:5]}

        try:
            payload["detail"] = self.detail(best["id"])
        except GraphNotFoundError:
            pass

        if best["kind"] == NODE_FILE:
            try:
                content = self.content(best["id"])
                # Excerpt only: whole files blow the model's context and the
                # retriever already supplies the relevant regions.
                payload["excerpt"] = "\n".join(content["content"].splitlines()[:120])
                payload["line_count"] = content["line_count"]
                payload["redacted"] = content["redacted"]
            except GraphNotFoundError:
                pass

        logger.info("inspect_file(%r) -> %s", term, best["id"])
        return payload
