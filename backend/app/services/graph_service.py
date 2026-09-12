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
    EDGE_CALLS,
    EDGE_CONTAINS,
    EDGE_DEPENDS_ON,
    EDGE_IMPLEMENTS,
    EDGE_IMPORTS,
    EDGE_INHERITS,
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

#: Nodes returned by an explicit set request. A flow is at most a few dozen
#: nodes; anything larger is a caller trying to fetch the graph one list at a
#: time, which is what the level-by-level design exists to prevent.
MAX_NODE_SET = 60

SYMBOL_KINDS = (NODE_CLASS, NODE_FUNCTION, NODE_METHOD)
CONTAINER_KINDS = (NODE_MODULE, NODE_DIRECTORY, NODE_FILE)

#: Relationships drawn between nodes on screen. CONTAINS is left out because
#: the hierarchy is already the layout.
RELATION_EDGES = [EDGE_IMPORTS, EDGE_DEPENDS_ON, EDGE_CALLS, EDGE_INHERITS, EDGE_IMPLEMENTS]

#: What the knowledge graph draws: files and the symbols inside them.
#: Repository, module and directory nodes are the folder tree, not knowledge;
#: external packages carry no edges, so they would only float unconnected.
KNOWLEDGE_KINDS = (NODE_FILE, NODE_CLASS, NODE_FUNCTION, NODE_METHOD)

#: Nodes in one knowledge graph response. FastAPI has ~5,800 files and
#: symbols; past this a force layout stops being readable, so the most
#: connected nodes are kept and the caller is told the rest were withheld.
DEFAULT_KNOWLEDGE_LIMIT = 1500

#: Node metadata the knowledge graph's tooltips and selection card show.
KNOWLEDGE_NODE_DATA = ("language", "line_count", "relative_path", "start_line", "end_line")


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

    def knowledge_graph(
        self, repository: str, limit: int = DEFAULT_KNOWLEDGE_LIMIT
    ) -> dict:
        """The whole repository as one graph of files, symbols and relationships.

        The exception to reading one level at a time: the knowledge graph is
        drawn all at once, so it is served all at once — but bounded. When a
        repository has more than ``limit`` files and symbols, the most
        connected are kept, because an isolated node adds a dot and no
        knowledge. A symbol is only ever kept together with the file (and
        class) that contains it, so nothing appears detached from its source.
        """
        if not self.store.has_repository(repository):
            raise GraphNotFoundError(f"Repository {repository!r} has no graph.")

        candidates = {
            n["id"]: n for n in self.store.nodes_of_kinds(repository, KNOWLEDGE_KINDS)
        }
        relations = [
            edge
            for edge in self.store.edges_of_kinds(repository, RELATION_EDGES)
            if edge["source"] in candidates
            and edge["target"] in candidates
            and edge["source"] != edge["target"]
        ]

        degree: dict[str, int] = {}
        for edge in relations:
            degree[edge["source"]] = degree.get(edge["source"], 0) + 1
            degree[edge["target"]] = degree.get(edge["target"], 0) + 1

        # Connected before isolated, then most connected first; files win ties
        # so the skeleton survives truncation.
        ranked = sorted(
            candidates.values(),
            key=lambda n: (
                degree.get(n["id"], 0) == 0,
                -degree.get(n["id"], 0),
                n["kind"] != NODE_FILE,
                n["key"],
            ),
        )

        selected: dict[str, dict] = {}
        for node in ranked:
            if len(selected) >= limit:
                break
            # The node plus any containers not yet selected, up to its file.
            chain: list[dict] = []
            current: dict | None = node
            while current is not None and current["id"] not in selected:
                chain.append(current)
                if current["kind"] == NODE_FILE:
                    break
                current = candidates.get(current["parent_id"] or "")
            if len(selected) + len(chain) > limit:
                continue
            for member in chain:
                selected[member["id"]] = member

        # Edges and nodes are trimmed to what the drawing and the chat use: at
        # FastAPI's size the full rows are 2.2 MB, most of it unused metadata.
        edges = [
            {"kind": edge["kind"], "source": edge["source"], "target": edge["target"]}
            for edge in relations
            if edge["source"] in selected and edge["target"] in selected
        ]
        # Containment is what pulls a file's symbols into a cluster around it.
        # Derived from parent links rather than read from CONTAINS rows: every
        # selected symbol's parent is already in memory.
        edges.extend(
            {"kind": EDGE_CONTAINS, "source": node["parent_id"], "target": node["id"]}
            for node in selected.values()
            if node["kind"] != NODE_FILE and node["parent_id"] in selected
        )

        return {
            "repository": repository,
            "nodes": [
                {
                    "id": node["id"],
                    "kind": node["kind"],
                    "name": node["name"],
                    "key": node["key"],
                    "parent_id": node["parent_id"],
                    "data": {
                        k: node["data"][k]
                        for k in KNOWLEDGE_NODE_DATA
                        if node["data"].get(k) is not None
                    },
                }
                for node in selected.values()
            ],
            "edges": edges,
            "counts": self.store.counts(repository),
            "total_nodes": len(candidates),
            "truncated": len(selected) < len(candidates),
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
            for edge in self.store.outgoing(child["id"], kinds=RELATION_EDGES)
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

    def node_set(self, node_ids: list[str]) -> dict:
        """An arbitrary set of nodes, plus the edges among them.

        Every other read here is scoped to one level of the hierarchy, because
        that is how the Explorer draws. A *flow* is the exception: it crosses
        levels by nature — ``backend/main.py`` sits beside ``backend/routes/
        auth.py``, one level deeper — so expanding a common parent shows some
        of the chain and silently omits the rest.

        This is deliberately not a "load the graph" endpoint. The caller must
        already know exactly which nodes it wants, and the set is capped.
        """
        nodes = self.store.nodes_by_ids(node_ids[:MAX_NODE_SET])
        # Preserve the caller's order: a flow's steps are meaningful, and
        # SQLite's `IN` returns rows in storage order.
        position = {node_id: index for index, node_id in enumerate(node_ids)}
        nodes.sort(key=lambda n: position.get(n["id"], len(position)))

        present = {n["id"] for n in nodes}
        edges = [
            edge
            for node in nodes
            for edge in self.store.outgoing(node["id"], kinds=RELATION_EDGES)
            if edge["target"] in present
        ]
        return {"nodes": nodes, "edges": edges, "requested": len(node_ids)}

    def ancestors(self, node_id: str) -> list[dict]:
        """The chain from the repository root down to ``node_id``, root first.

        Exists for navigation. A [Explore this] click lands on a node the user
        has not drilled to, and the Explorer renders one level at a time — so
        it needs the whole breadcrumb, not just the node. Walking parents in
        the client would be one request per level; this is one request.
        """
        node = self.store.get_node(node_id)
        if node is None:
            raise GraphNotFoundError(f"Node {node_id!r} not found.")

        chain = [node]
        seen = {node_id}
        parent_id = node.get("parent_id")

        # Bounded and cycle-guarded: the hierarchy is five levels deep, and a
        # malformed parent link must not spin here.
        while parent_id and parent_id not in seen and len(chain) < 12:
            parent = self.store.get_node(parent_id)
            if parent is None:
                break
            seen.add(parent_id)
            chain.append(parent)
            parent_id = parent.get("parent_id")

        chain.reverse()
        return chain

    def dependencies(self, node_id: str) -> dict:
        """What this node depends on, and what depends on it.

        Returned together because the graph highlights both directions at
        once — the visual answer to "what depends on this?".
        """
        node = self.store.get_node(node_id)
        if node is None:
            raise GraphNotFoundError(f"Node {node_id!r} not found.")

        outgoing = self.store.outgoing(node_id, kinds=RELATION_EDGES)
        incoming = self.store.incoming(node_id, kinds=RELATION_EDGES)

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

    def commit_timeline(self, repository: str, limit: int = 60) -> dict:
        """Commits as dots on a line — the Part 11 timeline.

        Returned oldest-first because the UI draws left to right, while the
        store returns newest-first (the natural order for "recent commits").
        Reversing once here keeps that decision out of the component.

        ``lanes`` from :meth:`timeline` is deliberately *not* merged in. Module
        activity and commit dots answer different questions, and the frontend
        shows them in different places; joining them here would force one to
        change whenever the other did.
        """
        if not self.store.has_repository(repository):
            raise GraphNotFoundError(f"Repository {repository!r} has no graph.")

        commits = self.store.commits(repository, limit=limit)
        releases = {
            ref["sha"]: ref["name"]
            for ref in self.store.refs(repository, "tag")
            if ref.get("sha")
        }

        dots = [
            {
                "sha": commit["sha"],
                "short_sha": commit["sha"][:8],
                "author": commit["author"] or "unknown",
                "authored_at": commit["authored_at"],
                "summary": commit["summary"] or "",
                "files_changed": commit["file_count"],
                "is_merge": bool(commit["is_merge"]),
                # Merge dots branch off the trunk in the drawing; tagged
                # commits get a release marker.
                "release": releases.get(commit["sha"], ""),
            }
            for commit in reversed(commits)
        ]

        return {
            "repository": repository,
            "commits": dots,
            "range": self.store.commit_range(repository),
            "truncated": len(commits) >= limit,
        }

    def commit_detail(self, repository: str, sha: str) -> dict:
        """One commit with the files it touched, resolved onto graph nodes.

        Resolving to node IDs is what makes a commit clickable: the frontend
        can select the changed files in the graph without a second round trip
        per file. Paths with no node are still listed — history covers files
        the parser excludes, and hiding them would misreport the commit.
        """
        commit = self.store.commit(repository, sha)
        if commit is None:
            raise GraphNotFoundError(f"Commit {sha!r} is not in the indexed history.")

        files = self.store.files_in_commit(repository, commit["sha"])
        nodes = {
            node["key"]: node["id"]
            for node in self.store.nodes_by_ids(
                [f"{repository}::{NODE_FILE}::{f['relative_path']}" for f in files]
            )
        }

        return {
            "repository": repository,
            "commit": {
                **commit,
                "short_sha": commit["sha"][:8],
                "is_merge": bool(commit["is_merge"]),
            },
            "files": [
                {
                    "relative_path": f["relative_path"],
                    "module": f["module"] or "",
                    "change_type": f["change_type"] or "M",
                    "node_id": nodes.get(f["relative_path"], ""),
                    "indexed": f["relative_path"] in nodes,
                }
                for f in files
            ],
            "files_changed": len(files),
        }

    def snapshot_at_commit(self, repository: str, sha: str) -> dict:
        """Repository state as of a commit — the timeline's click target.

        Delegates to the timestamp-based :meth:`snapshot` rather than
        reconstructing a tree. Same approximation, same caveat, and the commit
        is echoed back so the UI can label what it is showing.
        """
        commit = self.store.commit(repository, sha)
        if commit is None:
            raise GraphNotFoundError(f"Commit {sha!r} is not in the indexed history.")

        result = self.snapshot(repository, int(commit["authored_at"]))
        result["commit"] = {
            **commit,
            "short_sha": commit["sha"][:8],
            "is_merge": bool(commit["is_merge"]),
        }
        return result

    def changes_between(self, repository: str, start_sha: str, end_sha: str) -> dict:
        """What changed between two commits (Part 12).

        Both endpoints are resolved to timestamps and the window is scanned
        forward, so the arguments can be given in either order.
        """
        start = self.store.commit(repository, start_sha)
        end = self.store.commit(repository, end_sha)
        if start is None or end is None:
            raise GraphNotFoundError("One or both commits are not in the indexed history.")

        low, high = sorted((int(start["authored_at"]), int(end["authored_at"])))
        changes = self.store.changes_between(repository, low, high)

        modules: dict[str, int] = {}
        for change in changes:
            module = change["module"] or "(root)"
            modules[module] = modules.get(module, 0) + change["commits"]

        return {
            "repository": repository,
            "from": {**start, "short_sha": start["sha"][:8]},
            "to": {**end, "short_sha": end["sha"][:8]},
            "files": changes[:60],
            "files_touched": len(changes),
            "modules": sorted(modules.items(), key=lambda kv: -kv[1]),
        }

    # -- insights ----------------------------------------------------------

    def hotspots(self, repository: str, limit: int = 8) -> list[dict]:
        """Structural, churn and complexity hotspots, each with its evidence."""
        if not self.store.has_repository(repository):
            raise GraphNotFoundError(f"Repository {repository!r} has no graph.")
        from app.graph.insights import InsightAnalyzer

        return [h.to_dict() for h in InsightAnalyzer(self.store).hotspots(repository, limit)]

    def flow(self, repository: str, start_node_id: str | None = None) -> dict:
        """A static path through the import graph from an entry point."""
        if not self.store.has_repository(repository):
            raise GraphNotFoundError(f"Repository {repository!r} has no graph.")
        from app.graph.insights import InsightAnalyzer

        return InsightAnalyzer(self.store).flow(repository, start_node_id).to_dict()

    def neighborhood(self, node_id: str, hops: int = 1, limit: int = 40) -> dict:
        """Nodes and edges within ``hops`` of one node, for focused rendering.

        This is what an [Explore this] click loads: enough of the graph to
        show why the answer named this node, and no more. It is the same
        traversal the agent uses to build context, exposed so the picture and
        the prose come from one source.
        """
        from app.rag.graph_retriever import GraphRetriever

        node = self.store.get_node(node_id)
        if node is None:
            raise GraphNotFoundError(f"Node {node_id!r} not found.")

        context = GraphRetriever(self, node_budget=limit, hops=hops).retrieve(
            repository=node["repository"], focus_node_id=node_id, hops=hops
        )
        return {
            "node": node,
            "nodes": [hit.node for hit in context.hits],
            "edges": context.edges,
            "seeds": context.seeds,
            "truncated": bool(context.withheld),
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
