"""Graph model, analysis, building and persistence (Phase 3, stages 3.2-3.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.graph.analyzer import analyse, analyse_python
from app.graph.builder import GraphBuilder, directory_of, module_of
from app.graph.model import (
    EDGE_CONTAINS,
    EDGE_DEPENDS_ON,
    EDGE_IMPORTS,
    EDGE_INHERITS,
    NODE_CLASS,
    NODE_FILE,
    NODE_METHOD,
    NODE_MODULE,
    Edge,
    Node,
    RepositoryGraph,
    node_id,
    split_node_id,
)
from app.graph.store import GraphStore

PY_SOURCE = '''\
"""Docstring."""
import os
from pathlib import Path
from .helpers import assist
from app.services.other import Other


class Base:
    def shared(self):
        return 1


class Service(Base):
    """A service."""

    def __init__(self, name):
        self.name = name

    async def run(self):
        return assist(self.name)


def standalone(value):
    return Service(value)
'''


def parsed(path: str, content: str, language: str = "python") -> dict:
    return {
        "repository": "demo",
        "relative_path": path,
        "file_name": path.split("/")[-1],
        "language": language,
        "extension": "." + path.rsplit(".", 1)[-1],
        "file_hash": "abc123",
        "content": content,
    }


# ------------------------------------------------------------------- model


def test_node_id_round_trips():
    identifier = node_id("demo", NODE_FILE, "app/main.py")
    assert split_node_id(identifier) == ("demo", NODE_FILE, "app/main.py")


def test_node_id_is_deterministic():
    assert node_id("demo", NODE_FILE, "a.py") == node_id("demo", NODE_FILE, "a.py")


def test_malformed_node_id_raises():
    with pytest.raises(ValueError):
        split_node_id("nonsense")


def test_duplicate_edges_are_collapsed():
    graph = RepositoryGraph(repository="demo")
    edge = Edge("demo", EDGE_IMPORTS, "a", "b")
    assert graph.add_edge(edge) is not None
    assert graph.add_edge(Edge("demo", EDGE_IMPORTS, "a", "b")) is None
    assert len(graph.edges) == 1


def test_node_level_reflects_hierarchy():
    repo = Node("i", "demo", "repository", "demo", "demo")
    file = Node("i", "demo", NODE_FILE, "a.py", "a.py")
    method = Node("i", "demo", NODE_METHOD, "m", "a.py::C.m")
    assert repo.level < file.level < method.level


# ---------------------------------------------------------------- modules


@pytest.mark.parametrize(
    "path,module",
    [
        ("fastapi/routing.py", "fastapi"),
        ("tests/test_x.py", "tests"),
        ("src/auth/service.py", "src/auth"),
        ("app/rag/parser.py", "app/rag"),
        ("lib/core/util.py", "lib/core"),
        ("setup.py", "(root)"),
    ],
)
def test_module_assignment(path: str, module: str):
    assert module_of(path) == module


def test_directory_of():
    assert directory_of("a/b/c.py") == "a/b"
    assert directory_of("c.py") == ""


# --------------------------------------------------------------- analysis


def test_python_classes_and_methods():
    analysis = analyse_python(PY_SOURCE, "app/services/service.py")
    assert [c.name for c in analysis.classes] == ["Base", "Service"]
    assert {m.qualified_name for m in analysis.methods} == {
        "Base.shared", "Service.__init__", "Service.run"
    }
    assert [f.name for f in analysis.functions] == ["standalone"]


def test_python_records_base_classes():
    analysis = analyse_python(PY_SOURCE, "a.py")
    service = next(c for c in analysis.classes if c.name == "Service")
    assert service.bases == ["Base"]


def test_python_marks_async():
    analysis = analyse_python(PY_SOURCE, "a.py")
    assert next(m for m in analysis.methods if m.name == "run").is_async


def test_python_captures_calls():
    analysis = analyse_python(PY_SOURCE, "a.py")
    assert "assist" in next(m for m in analysis.methods if m.name == "run").calls


def test_python_imports_absolute_and_relative():
    analysis = analyse_python(PY_SOURCE, "a.py")
    modules = {i.module: i for i in analysis.imports}
    assert "os" in modules
    assert modules["helpers"].relative_level == 1
    assert modules["app.services.other"].names == ["Other"]


def test_syntax_error_degrades_without_raising():
    analysis = analyse_python("def broken(:\n    pass\n", "bad.py")
    assert analysis.parse_failed
    assert analysis.symbols == []


def test_unknown_language_is_safe():
    analysis = analyse("body", "a.zzz", "brainfuck")
    assert analysis.symbols == []
    assert not analysis.parse_failed


def test_javascript_symbols_and_imports():
    source = (
        "import { JwtService } from './jwt.service';\n"
        "const fs = require('fs');\n"
        "export class AuthService extends Base {\n}\n"
        "export function login() {}\n"
    )
    analysis = analyse(source, "src/auth/auth.service.js", "javascript")
    assert any(c.name == "AuthService" for c in analysis.classes)
    assert any(f.name == "login" for f in analysis.functions)
    assert any("jwt.service" in i.module for i in analysis.imports)


def test_complexity_band():
    small = analyse("x = 1\n", "a.py", "python")
    big = analyse("\n".join(f"def f{i}(): pass" for i in range(60)), "b.py", "python")
    assert small.complexity_band() == "low"
    assert big.complexity_band() == "high"


# --------------------------------------------------------------- building


@pytest.fixture
def built() -> GraphBuilder:
    builder = GraphBuilder("demo")
    builder.build(
        [
            parsed("app/services/service.py", PY_SOURCE),
            parsed("app/services/helpers.py", "def assist(name):\n    return name\n"),
            parsed("app/services/other.py", "class Other:\n    pass\n"),
            # Genuinely nested, so a `directory` node is created rather than
            # collapsed into its module.
            parsed("app/services/nested/deep.py", "def deep():\n    return 1\n"),
            parsed("tests/test_service.py", "from app.services.service import Service\n"),
        ]
    )
    return builder


def test_hierarchy_is_built(built: GraphBuilder):
    kinds = {n.kind for n in built.graph.nodes}
    assert {"repository", "module", "directory", "file", "class", "function", "method"} <= kinds


def test_files_are_contained_by_directories(built: GraphBuilder):
    contains = [e for e in built.graph.edges if e.kind == EDGE_CONTAINS]
    assert contains
    file_node = next(n for n in built.graph.nodes if n.key == "app/services/service.py")
    assert file_node.parent_id is not None


def test_file_metadata_is_recorded(built: GraphBuilder):
    node = next(n for n in built.graph.nodes if n.key == "app/services/service.py")
    assert node.data["language"] == "python"
    assert node.data["class_count"] == 2
    assert node.data["line_count"] > 10
    assert node.data["complexity"] in {"low", "medium", "high"}


def test_relative_import_resolves(built: GraphBuilder):
    imports = [e for e in built.graph.edges if e.kind == EDGE_IMPORTS]
    targets = {e.target_id for e in imports}
    assert node_id("demo", NODE_FILE, "app/services/helpers.py") in targets


def test_absolute_import_resolves(built: GraphBuilder):
    imports = [e for e in built.graph.edges if e.kind == EDGE_IMPORTS]
    targets = {e.target_id for e in imports}
    assert node_id("demo", NODE_FILE, "app/services/other.py") in targets


def test_cross_module_dependency_edge(built: GraphBuilder):
    depends = [e for e in built.graph.edges if e.kind == EDGE_DEPENDS_ON]
    # tests/ imports app/services -> a module-level DEPENDS_ON edge.
    assert any(e.source_id == node_id("demo", NODE_MODULE, "tests") for e in depends)


def test_inherits_edge_within_a_file(built: GraphBuilder):
    inherits = [e for e in built.graph.edges if e.kind == EDGE_INHERITS]
    assert any(e.target_id.endswith("::Base") for e in inherits)


def test_methods_hang_off_their_class(built: GraphBuilder):
    service = next(n for n in built.graph.nodes if n.key.endswith("::Service") and n.kind == NODE_CLASS)
    methods = [n for n in built.graph.nodes if n.kind == NODE_METHOD and n.parent_id == service.id]
    assert {m.name for m in methods} == {"Service.__init__", "Service.run"}


def test_external_packages_need_repeat_use():
    builder = GraphBuilder("demo")
    builder.build(
        [
            parsed("a.py", "import requests\nimport onlyonce\n"),
            parsed("b.py", "import requests\n"),
        ]
    )
    externals = {n.name for n in builder.graph.nodes if n.kind == "external"}
    assert "requests" in externals
    assert "onlyonce" not in externals


def test_build_stats(built: GraphBuilder):
    assert built.stats.files == 5
    assert built.stats.symbols > 0
    assert built.stats.internal_imports >= 2


# ------------------------------------------------------------------ store


@pytest.fixture
def store(tmp_path: Path) -> GraphStore:
    return GraphStore(path=str(tmp_path / "graph.sqlite3"))


def test_store_round_trip(store: GraphStore, built: GraphBuilder):
    store.replace_repository("demo", built.graph.nodes, built.graph.edges)
    assert store.has_repository("demo")
    assert store.repositories() == ["demo"]

    node = store.find_node("demo", "app/services/service.py", NODE_FILE)
    assert node is not None
    assert node["data"]["language"] == "python"


def test_store_rebuild_removes_stale_nodes(store: GraphStore, built: GraphBuilder):
    store.replace_repository("demo", built.graph.nodes, built.graph.edges)
    before = store.counts("demo")["file"]

    smaller = GraphBuilder("demo")
    smaller.build([parsed("app/only.py", "x = 1\n")])
    store.replace_repository("demo", smaller.graph.nodes, smaller.graph.edges)

    assert store.counts("demo")["file"] == 1 < before


def test_store_children(store: GraphStore, built: GraphBuilder):
    store.replace_repository("demo", built.graph.nodes, built.graph.edges)
    root = store.roots("demo", "repository")[0]
    modules = store.children(root["id"], kinds=[NODE_MODULE])
    assert {m["name"] for m in modules} >= {"app/services", "tests"}


def test_store_dependency_queries(store: GraphStore, built: GraphBuilder):
    store.replace_repository("demo", built.graph.nodes, built.graph.edges)
    helpers = node_id("demo", NODE_FILE, "app/services/helpers.py")
    dependents = store.incoming(helpers, kinds=[EDGE_IMPORTS])
    assert dependents
    assert any("service.py" in e["source"] for e in dependents)


def test_store_search(store: GraphStore, built: GraphBuilder):
    store.replace_repository("demo", built.graph.nodes, built.graph.edges)
    hits = store.search_nodes("demo", "Service")
    assert any(h["name"].endswith("Service") for h in hits)


def test_store_delete(store: GraphStore, built: GraphBuilder):
    store.replace_repository("demo", built.graph.nodes, built.graph.edges)
    store.delete_repository("demo")
    assert not store.has_repository("demo")


def test_store_isolates_repositories(store: GraphStore, built: GraphBuilder):
    store.replace_repository("demo", built.graph.nodes, built.graph.edges)
    other = GraphBuilder("other")
    other.build([parsed("x.py", "y = 2\n")])
    store.replace_repository("other", other.graph.nodes, other.graph.edges)

    assert set(store.repositories()) == {"demo", "other"}
    assert store.counts("other")["file"] == 1


# ---------------------------------------------------------------- history


def test_history_round_trip(store: GraphStore):
    commits = [
        {"sha": "a1", "author": "Ada", "authored_at": 1_700_000_000, "summary": "add auth", "is_merge": False},
        {"sha": "b2", "author": "Ada", "authored_at": 1_700_100_000, "summary": "merge", "is_merge": True},
    ]
    files = [
        {"sha": "a1", "relative_path": "auth/service.py", "module": "auth",
         "change_type": "A", "authored_at": 1_700_000_000},
    ]
    refs = [{"name": "main", "kind": "branch", "sha": "b2", "created_at": 1_700_100_000}]

    store.replace_history("demo", commits, files, refs)

    assert store.commit_range("demo")["total"] == 2
    assert store.modules_first_seen("demo") == {"auth": 1_700_000_000}
    assert [r["name"] for r in store.refs("demo", "branch")] == ["main"]
    assert store.commits_for_path("demo", "auth/service.py")[0]["sha"] == "a1"


def test_history_replace_is_not_additive(store: GraphStore):
    commit = {"sha": "a1", "author": "A", "authored_at": 1, "summary": "s", "is_merge": False}
    store.replace_history("demo", [commit], [], [])
    store.replace_history("demo", [commit], [], [])
    assert store.commit_range("demo")["total"] == 1


def test_paths_existing_at_is_time_filtered(store: GraphStore):
    files = [
        {"sha": "a", "relative_path": "old.py", "module": "m", "change_type": "A", "authored_at": 100},
        {"sha": "b", "relative_path": "new.py", "module": "m", "change_type": "A", "authored_at": 500},
    ]
    store.replace_history("demo", [], files, [])
    assert store.paths_existing_at("demo", 200) == {"old.py"}
    assert store.paths_existing_at("demo", 900) == {"old.py", "new.py"}


def test_timeline_buckets_group_by_module_and_month(store: GraphStore):
    files = [
        {"sha": "a", "relative_path": "auth/x.py", "module": "auth", "change_type": "A", "authored_at": 1_700_000_000},
        {"sha": "b", "relative_path": "auth/y.py", "module": "auth", "change_type": "M", "authored_at": 1_700_000_500},
        {"sha": "c", "relative_path": "api/z.py", "module": "api", "change_type": "A", "authored_at": 1_700_001_000},
    ]
    store.replace_history("demo", [], files, [])
    buckets = {(b["module"], b["period"]): b for b in store.timeline_buckets("demo")}
    assert len(buckets) == 2
    auth = next(v for k, v in buckets.items() if k[0] == "auth")
    assert auth["commits"] == 2
