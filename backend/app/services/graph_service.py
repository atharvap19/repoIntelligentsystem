"""Query layer over the repository graph.

Everything the API and the agent need to read the graph lives here, so both
consume the same shapes and neither talks to SQLite directly.

The guiding constraint is Part 17: never return the whole repository. FastAPI's
graph has 6,363 nodes; rendering that at once would be unusable and slow.
Every read is scoped to one level of the hierarchy, and the frontend expands
by asking for a node's children.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.graph.model import (
    EDGE_CONTAINS,
    EDGE_DEPENDS_ON,
    EDGE_IMPORTS,
    NODE_CLASS,
    NODE_DIRECTORY,
    NODE_EXTERNAL,
    NODE_FILE,
    NODE_FUNCTION,
    NODE_METHOD,
    NODE_MODULE,
    NODE_REPOSITORY,
    split_node_id,
)
from app.graph.store import GraphStore
from app.utils.redaction import redact

logger = logging.getLogger(__name__)

#: Children returned per expansion. A directory with 400 files is truncated
#: rather than rendered — the caller is told how many were withheld.
DEFAULT_CHILD_LIMIT = 120

#: Bytes of file content served in one response.
MAX_CONTENT_BYTES = 400_000

SYMBOL_KINDS = (NODE_CLASS, NODE_FUNCTION, NODE_METHOD)
CONTAINER_KINDS = (NODE_MODULE, NODE_DIRECTORY, NODE_FILE)


class GraphNotFoundError(LookupError):
    """Raised when a repository or node is not in the graph."""


class GraphService:
    """Read-side operations on the repository graph."""

    def __init__(
        self,
        graph_store: GraphStore | None = None,
        repositories_root: str = "repositories",
    ):
        self.store = graph_store or GraphStore()
        self.repositories_root = Path(repositories_root)

    # -- repository level --------------------------------------------------

    def repositories(self) -> list[dict]:
        # `counts` is nested rather than splatted: it contains a `repository`
        # key of its own (the count of repository-kind nodes) which would
        # overwrite the name with the integer 1.
        return [
            {"repository": name, "counts": self.store.counts(name)}
            for name in self.store.repositories()
        ]

    def overview(self, repository: str) -> dict:
        """The initial view: the repository, its modules, and how they depend.

        This is the only call the frontend makes on load. Files and symbols are
        summarised as counts, never enumerated.
        """
        if not self.store.has_repository(repository):
            raise GraphNotFoundError(f"Repository {repository!r} has no graph.")

        roots = self.store.roots(repository, NODE_REPOSITORY)
        if not roots:
            raise GraphNotFoundError(f"Repository {repository!r} has no root node.")
        root = roots[0]

        modules = self.store.children(root["id"], kinds=[NODE_MODULE])
        first_seen = self.store.modules_first_seen(repository)

        for module in modules:
            module["data"] = {
                **module["data"],
                **self._module_rollup(module["id"]),
                "first_seen": first_seen.get(module["name"]),
            }

        externals = self.store.children(root["id"], kinds=[NODE_EXTERNAL])
        module_ids = {m["id"] for m in modules}
        edges = [
            edge
            for module in modules
            for edge in self.store.outgoing(module["id"], kinds=[EDGE_DEPENDS_ON])
            if edge["target"] in module_ids
        ]

        return {
            "repository": repository,
            "root": root,
            "nodes": modules + externals,
            "edges": edges,
            "counts": self.store.counts(repository),
            "timeline": self.store.commit_range(repository),
        }

    def _module_rollup(self, module_id: str) -> dict:
        """Aggregate a module's subtree so it can be labelled without expanding."""
        files = 0
        lines = 0
        stack = [module_id]
        languages: dict[str, int] = {}

        while stack:
            current = stack.pop()
            for child in self.store.children(current):
                if child["kind"] == NODE_FILE:
                    files += 1
                    lines += int(child["data"].get("line_count") or 0)
                    language = child["data"].get("language")
                    if language:
                        languages[language] = languages.get(language, 0) + 1
                elif child["kind"] in (NODE_DIRECTORY, NODE_MODULE):
                    stack.append(child["id"])

        return {
            "file_count": files,
            "line_count": lines,
            "languages": languages,
            "primary_language": max(languages, key=languages.get) if languages else "",
        }

    # -- expansion ---------------------------------------------------------

    def expand(
        self,
        node_id: str,
        limit: int = DEFAULT_CHILD_LIMIT,
        include_symbols: bool = True,
    ) -> dict:
        """One level of children for a node, plus edges among them.

        The hierarchy is repository -> module -> directory -> file -> symbol.
        Expanding a file yields its classes and functions; expanding a class
        yields its methods.
        """
        node = self.store.get_node(node_id)
        if node is None:
            raise GraphNotFoundError(f"Node {node_id!r} not found.")

        kinds: list[str] | None = None
        if node["kind"] == NODE_REPOSITORY:
            kinds = [NODE_MODULE]
        elif node["kind"] in (NODE_MODULE, NODE_DIRECTORY):
            kinds = [NODE_DIRECTORY, NODE_FILE]
        elif node["kind"] == NODE_FILE and not include_symbols:
            kinds = []

        children = self.store.children(node_id, kinds=kinds) if kinds != [] else []
        total = len(children)

        # Containers first, then largest symbols — the useful things are at the
        # top when the list is truncated.
        children.sort(
            key=lambda c: (
                CONTAINER_KINDS.index(c["kind"]) if c["kind"] in CONTAINER_KINDS else 9,
                -int(c["data"].get("line_count") or 0),
                c["name"],
            )
        )
        truncated = children[:limit]
        child_ids = {c["id"] for c in truncated}

        edges = [
            edge
            for child in truncated
            for edge in self.store.outgoing(child["id"], kinds=[EDGE_IMPORTS, EDGE_DEPENDS_ON])
            if edge["target"] in child_ids
        ]

        return {
            "node": node,
            "nodes": truncated,
            "edges": edges,
            "total_children": total,
            "truncated": total > len(truncated),
        }

    # -- node detail -------------------------------------------------------

    def node_detail(self, node_id: str) -> dict:
        """File- or symbol-level intelligence for the details panel."""
        node = self.store.get_node(node_id)
        if node is None:
            raise GraphNotFoundError(f"Node {node_id!r} not found.")

        repository, kind, key = split_node_id(node_id)
        detail: dict[str, Any] = {"node": node}

        if kind == NODE_FILE:
            symbols = self.store.children(node_id, kinds=list(SYMBOL_KINDS))
            detail["symbols"] = symbols
            detail["dependencies"] = self._resolve(
                self.store.outgoing(node_id, kinds=[EDGE_IMPORTS]), "target"
            )
            detail["dependents"] = self._resolve(
                self.store.incoming(node_id, kinds=[EDGE_IMPORTS]), "source"
            )
            detail["history"] = self.store.commits_for_path(repository, key, limit=10)
        elif kind in SYMBOL_KINDS:
            detail["children"] = self.store.children(node_id)
            parent = self.store.get_node(node["parent_id"]) if node["parent_id"] else None
            detail["parent"] = parent
            relative_path = node["data"].get("relative_path", "")
            if relative_path:
                detail["history"] = self.store.commits_for_path(
                    repository, relative_path, limit=10
                )
        else:
            detail["children_count"] = len(self.store.children(node_id))

        return detail

    def dependencies(self, node_id: str) -> dict:
        """What this node depends on, and what depends on it.

        Returned together because the graph highlights both directions at
        once — the visual answer to "what depends on this?".
        """
        node = self.store.get_node(node_id)
        if node is None:
            raise GraphNotFoundError(f"Node {node_id!r} not found.")

        kinds = [EDGE_IMPORTS, EDGE_DEPENDS_ON]
        outgoing = self.store.outgoing(node_id, kinds=kinds)
        incoming = self.store.incoming(node_id, kinds=kinds)

        return {
            "node": node,
            "dependencies": self._resolve(outgoing, "target"),
            "dependents": self._resolve(incoming, "source"),
            "edges": outgoing + incoming,
        }

    def _resolve(self, edges: list[dict], side: str) -> list[dict]:
        """Turn edge endpoints into full nodes, preserving edge metadata."""
        ids = [edge[side] for edge in edges]
        by_id = {n["id"]: n for n in self.store.nodes_by_ids(ids)}
        resolved = []
        for edge in edges:
            node = by_id.get(edge[side])
            if node:
                resolved.append({**node, "via": edge["kind"], "edge_data": edge["data"]})
        return resolved

    # -- content -----------------------------------------------------------

    def file_content(self, node_id: str, repositories_root: Path | None = None) -> dict:
        """Read a file from disk for the code cell.

        Content is read live rather than stored in the graph: it can be large,
        it is already on disk, and duplicating it would double the index. It
        passes through redaction before leaving the backend.
        """
        node = self.store.get_node(node_id)
        if node is None or node["kind"] != NODE_FILE:
            raise GraphNotFoundError(f"{node_id!r} is not a file node.")

        repository, _, relative_path = split_node_id(node_id)
        root = Path(
            (self.store.get_meta(repository, "path") or {}).get("value")
            or (repositories_root or self.repositories_root) / repository
        )

        target = (root / relative_path).resolve()
        # Refuse to serve anything outside the checkout, whatever the node says.
        if not str(target).startswith(str(root.resolve())):
            raise GraphNotFoundError("Refusing to read outside the repository root.")
        if not target.is_file():
            raise GraphNotFoundError(f"File not found on disk: {relative_path}")

        raw = target.read_bytes()[:MAX_CONTENT_BYTES]
        text = raw.decode("utf-8", errors="replace")
        result = redact(text)

        return {
            "node": node,
            "relative_path": relative_path,
            "language": node["data"].get("language", ""),
            "content": result.text,
            "line_count": result.text.count("\n") + 1,
            "truncated": target.stat().st_size > MAX_CONTENT_BYTES,
            "redacted": result.redacted,
            "redaction_findings": result.findings,
            "symbols": self.store.children(node_id, kinds=list(SYMBOL_KINDS)),
        }

    # -- search / history --------------------------------------------------

    def search(self, repository: str, term: str, limit: int = 25) -> list[dict]:
        if not term.strip():
            return []
        return self.store.search_nodes(repository, term.strip(), limit=limit)

    def history(self, node_id: str, limit: int = 20) -> list[dict]:
        node = self.store.get_node(node_id)
        if node is None:
            raise GraphNotFoundError(f"Node {node_id!r} not found.")
        repository, kind, key = split_node_id(node_id)
        path = key if kind == NODE_FILE else node["data"].get("relative_path", "")
        if not path:
            return []
        return self.store.commits_for_path(repository, path, limit=limit)

    # -- timeline ----------------------------------------------------------

    def timeline(self, repository: str) -> dict:
        """Module activity over time, plus releases — the timeline's data.

        Lanes are modules rather than git branches. A clone has one local
        branch and the branch graph says little about how a codebase grew;
        module activity says a great deal, which is the "higher-level
        repository evolution" the brief asks for.
        """
        if not self.store.has_repository(repository):
            raise GraphNotFoundError(f"Repository {repository!r} has no graph.")

        buckets = self.store.timeline_buckets(repository)
        first_seen = self.store.modules_first_seen(repository)

        lanes: dict[str, dict] = {}
        for bucket in buckets:
            lane = lanes.setdefault(
                bucket["module"],
                {
                    "module": bucket["module"],
                    "first_seen": first_seen.get(bucket["module"]),
                    "periods": [],
                    "total_commits": 0,
                },
            )
            lane["periods"].append(
                {
                    "period": bucket["period"],
                    "commits": bucket["commits"],
                    "changes": bucket["changes"],
                    "first_at": bucket["first_at"],
                    "last_at": bucket["last_at"],
                }
            )
            lane["total_commits"] += bucket["commits"]

        ordered = sorted(lanes.values(), key=lambda l: -l["total_commits"])

        return {
            "repository": repository,
            "range": self.store.commit_range(repository),
            "lanes": ordered,
            "releases": [
                r for r in self.store.refs(repository, "tag") if r.get("created_at")
            ][-40:],
            "branches": self.store.refs(repository, "branch"),
        }

    def snapshot(self, repository: str, timestamp: int) -> dict:
        """Approximate the repository as it stood at ``timestamp``.

        Approximate by design: a path counts as present from its first commit
        onward, and deletions are not modelled. Exact reconstruction means
        reading a tree per commit, which is far more expensive than the
        feature warrants. The shape below is what an exact implementation
        would also return, so it can be swapped in later.
        """
        if not self.store.has_repository(repository):
            raise GraphNotFoundError(f"Repository {repository!r} has no graph.")

        module_first_seen = self.store.modules_first_seen(repository)
        path_first_seen = self.store.paths_first_seen(repository)

        # Count indexed files only. Counting raw historical paths mixes
        # denominators — FastAPI's `docs` module has 8 indexed files but 1,555
        # paths in history, most of them markdown the parser excludes.
        files_by_module: dict[str, list[str]] = {}
        for file_node in self.store.nodes_of_kind(repository, NODE_FILE):
            files_by_module.setdefault(file_node["data"].get("module", ""), []).append(
                file_node["key"]
            )

        roots = self.store.roots(repository, NODE_REPOSITORY)
        root = roots[0] if roots else None
        modules = self.store.children(root["id"], kinds=[NODE_MODULE]) if root else []

        present: list[dict] = []
        for module in modules:
            appeared = module_first_seen.get(module["name"])
            if appeared is not None and appeared > timestamp:
                continue

            # A path with no observed commit predates the walked window, so it
            # already existed; only a *known* later first-commit excludes it.
            present_files = sum(
                1
                for path in files_by_module.get(module["name"], [])
                if path_first_seen.get(path, 0) <= timestamp
            )

            module["data"] = {
                **module["data"],
                **self._module_rollup(module["id"]),
                "first_seen": appeared,
                "file_count_at": present_files,
            }
            present.append(module)

        module_ids = {m["id"] for m in present}
        edges = [
            edge
            for module in present
            for edge in self.store.outgoing(module["id"], kinds=[EDGE_DEPENDS_ON])
            if edge["target"] in module_ids
        ]

        return {
            "repository": repository,
            "timestamp": timestamp,
            "root": root,
            "nodes": present,
            "edges": edges,
            "files_present": sum(int(m["data"].get("file_count_at") or 0) for m in present),
            "approximate": True,
        }
