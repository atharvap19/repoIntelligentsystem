"""Repository graph vocabulary.

Node and edge kinds are plain string constants rather than enums so they
survive JSON and SQLite round trips without conversion, and so a new kind can
be added without a migration.

IDs are deterministic and human-readable — ``fastapi::file::fastapi/routing.py``
— which means a rebuild produces identical IDs, the frontend can construct an
ID it wants to fetch, and debugging a graph query does not require a join.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# -- node kinds ------------------------------------------------------------

NODE_REPOSITORY = "repository"
NODE_MODULE = "module"
NODE_DIRECTORY = "directory"
NODE_FILE = "file"
NODE_CLASS = "class"
NODE_FUNCTION = "function"
NODE_METHOD = "method"
NODE_EXTERNAL = "external"  # a third-party package, not a file in this repo

NODE_KINDS: tuple[str, ...] = (
    NODE_REPOSITORY,
    NODE_MODULE,
    NODE_DIRECTORY,
    NODE_FILE,
    NODE_CLASS,
    NODE_FUNCTION,
    NODE_METHOD,
    NODE_EXTERNAL,
)

#: Depth in the drill-down hierarchy. The frontend renders one level at a time,
#: so a node's level decides when it becomes visible.
NODE_LEVEL: dict[str, int] = {
    NODE_REPOSITORY: 0,
    NODE_MODULE: 1,
    NODE_DIRECTORY: 2,
    NODE_FILE: 3,
    NODE_CLASS: 4,
    NODE_FUNCTION: 4,
    NODE_METHOD: 5,
    NODE_EXTERNAL: 1,
}

# -- edge kinds ------------------------------------------------------------

EDGE_CONTAINS = "CONTAINS"
EDGE_IMPORTS = "IMPORTS"
EDGE_DEPENDS_ON = "DEPENDS_ON"
EDGE_CALLS = "CALLS"
EDGE_INHERITS = "INHERITS"
EDGE_IMPLEMENTS = "IMPLEMENTS"
EDGE_MODIFIED_BY = "MODIFIED_BY"
EDGE_BELONGS_TO_BRANCH = "BELONGS_TO_BRANCH"
EDGE_MERGED_INTO = "MERGED_INTO"
EDGE_CREATED_BY = "CREATED_BY"

EDGE_KINDS: tuple[str, ...] = (
    EDGE_CONTAINS,
    EDGE_IMPORTS,
    EDGE_DEPENDS_ON,
    EDGE_CALLS,
    EDGE_INHERITS,
    EDGE_IMPLEMENTS,
    EDGE_MODIFIED_BY,
    EDGE_BELONGS_TO_BRANCH,
    EDGE_MERGED_INTO,
    EDGE_CREATED_BY,
)

ID_SEPARATOR = "::"


def node_id(repository: str, kind: str, key: str) -> str:
    """Deterministic node identifier.

    Same inputs always give the same ID, so re-indexing overwrites rather than
    duplicating — the same property the vector store relies on.
    """
    return ID_SEPARATOR.join((repository, kind, key))


def split_node_id(identifier: str) -> tuple[str, str, str]:
    """Inverse of :func:`node_id`. Raises ``ValueError`` on a malformed ID."""
    parts = identifier.split(ID_SEPARATOR, 2)
    if len(parts) != 3:
        raise ValueError(f"Malformed node id: {identifier!r}")
    return parts[0], parts[1], parts[2]


@dataclass
class Node:
    """A vertex in the repository graph."""

    id: str
    repository: str
    kind: str
    #: Short display name — ``routing.py``, ``APIRouter``.
    name: str
    #: Stable locator: a repo-relative path for files, a dotted path for symbols.
    key: str
    parent_id: str | None = None
    #: Free-form, JSON-serialised on write. Language, line counts, symbol
    #: counts, complexity — whatever an analyser produced.
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def level(self) -> int:
        return NODE_LEVEL.get(self.kind, 9)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "repository": self.repository,
            "kind": self.kind,
            "name": self.name,
            "key": self.key,
            "parent_id": self.parent_id,
            "level": self.level,
            "data": self.data,
        }


@dataclass
class Edge:
    """A directed relationship between two nodes."""

    repository: str
    kind: str
    source_id: str
    target_id: str
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.source_id}|{self.kind}|{self.target_id}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "repository": self.repository,
            "kind": self.kind,
            "source": self.source_id,
            "target": self.target_id,
            "data": self.data,
        }


@dataclass
class RepositoryGraph:
    """An in-memory graph, before it is persisted."""

    repository: str
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    _edge_ids: set[str] = field(default_factory=set, repr=False)

    def add_node(self, node: Node) -> Node:
        self.nodes.append(node)
        return node

    def add_edge(self, edge: Edge) -> Edge | None:
        """Add an edge, ignoring an exact repeat.

        A file that imports two names from the same module in separate
        statements yields the same (source, kind, target) twice. The store's
        primary key would collapse those anyway; deduplicating here keeps the
        in-memory count honest instead of reporting more edges than are saved.
        """
        if edge.id in self._edge_ids:
            return None
        self._edge_ids.add(edge.id)
        self.edges.append(edge)
        return edge

    def counts(self) -> dict[str, int]:
        from collections import Counter

        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            **{f"node_{k}": v for k, v in Counter(n.kind for n in self.nodes).items()},
            **{f"edge_{k}": v for k, v in Counter(e.kind for e in self.edges).items()},
        }
