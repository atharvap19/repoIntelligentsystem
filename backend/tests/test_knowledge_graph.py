"""Knowledge graph, one-step import, and chat-to-graph highlighting."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent.response import (
    MAX_HIGHLIGHT,
    UnknownRepositoryError,
    graph_highlight,
    resolve_repository,
)
from app.api import graph as graph_api
from app.graph.builder import GraphBuilder
from app.graph.store import GraphStore
from app.rag.vector_store import VectorStore
from app.services.github_service import InvalidRepositoryUrl, parse_github_url
from app.services.graph_service import KNOWLEDGE_KINDS, GraphNotFoundError, GraphService
from app.services.import_service import (
    STATE_CLONING,
    STATE_ERROR,
    STATE_READY,
    ImportJob,
    ImportService,
)
from app.services.indexing_service import IndexingService

from .conftest import write
from .test_graph_api import FILES
from .test_indexing_service import FakeEmbeddingService

AUTH_FILE = "demo::file::src/auth/service.py"


@pytest.fixture
def service(tmp_path: Path) -> GraphService:
    store = GraphStore(path=str(tmp_path / "graph.sqlite3"))
    builder = GraphBuilder("demo")
    builder.build(FILES)
    store.replace_repository("demo", builder.graph.nodes, builder.graph.edges)
    return GraphService(graph_store=store, repositories_root=str(tmp_path))


# --------------------------------------------------------- knowledge graph


def test_knowledge_graph_holds_files_and_symbols_only(service: GraphService):
    graph = service.knowledge_graph("demo")
    kinds = {n["kind"] for n in graph["nodes"]}
    assert kinds <= set(KNOWLEDGE_KINDS)
    assert {"file", "class"} <= kinds
    assert graph["truncated"] is False
    assert graph["total_nodes"] == len(graph["nodes"])


def test_knowledge_graph_spans_directories_in_one_response(service: GraphService):
    keys = {n["key"] for n in service.knowledge_graph("demo")["nodes"] if n["kind"] == "file"}
    assert {"src/auth/service.py", "src/api/routes.py", "tests/test_auth.py"} <= keys


def test_knowledge_graph_edges_join_nodes_it_returned(service: GraphService):
    graph = service.knowledge_graph("demo")
    ids = {n["id"] for n in graph["nodes"]}
    assert graph["edges"]
    assert all(e["source"] in ids and e["target"] in ids for e in graph["edges"])
    assert ("demo::file::src/api/routes.py", AUTH_FILE, "IMPORTS") in {
        (e["source"], e["target"], e["kind"]) for e in graph["edges"]
    }


def test_symbols_are_contained_by_their_file(service: GraphService):
    graph = service.knowledge_graph("demo")
    contains = {(e["source"], e["target"]) for e in graph["edges"] if e["kind"] == "CONTAINS"}
    auth_class = next(n for n in graph["nodes"] if n["name"] == "AuthService")
    assert (AUTH_FILE, auth_class["id"]) in contains


def test_truncation_keeps_the_most_connected_node(service: GraphService):
    # service.py imports helpers.py and is imported by two files: degree 3.
    graph = service.knowledge_graph("demo", limit=1)
    assert [n["id"] for n in graph["nodes"]] == [AUTH_FILE]
    assert graph["truncated"] is True


@pytest.mark.parametrize("limit", [2, 3, 5, 8])
def test_a_kept_symbol_always_brings_its_containers(service: GraphService, limit: int):
    graph = service.knowledge_graph("demo", limit=limit)
    ids = {n["id"] for n in graph["nodes"]}
    assert len(ids) <= limit
    for node in graph["nodes"]:
        if node["kind"] != "file":
            assert node["parent_id"] in ids


def test_knowledge_graph_of_unknown_repository_raises(service: GraphService):
    with pytest.raises(GraphNotFoundError):
        service.knowledge_graph("nope")


def test_http_knowledge_graph(service: GraphService):
    app = FastAPI()
    app.include_router(graph_api.router)
    original = graph_api.service
    graph_api.service = service
    try:
        client = TestClient(app)
        assert client.get("/graph/demo/knowledge").json()["nodes"]
        assert client.get("/graph/demo/knowledge", params={"limit": 10}).status_code == 422
        assert client.get("/graph/nope/knowledge").status_code == 404
    finally:
        graph_api.service = original


# ----------------------------------------------------------- highlighting


def test_highlight_orders_subject_first_and_skips_undrawable_kinds():
    hits = [SimpleNamespace(id="demo::function::src/auth/helpers.py::assist"),
            SimpleNamespace(id=AUTH_FILE)]
    result = {
        "focus_node_id": "demo::module::src/auth",  # modules are not drawn
        "graph_context": SimpleNamespace(seeds=[AUTH_FILE], hits=hits),
        "ui_action": {
            "type": "highlight_dependencies",
            "node_id": AUTH_FILE,
            "dependencies": ["demo::file::src/auth/helpers.py"],
            "dependents": ["demo::file::src/api/routes.py", "demo::external::pydantic"],
        },
        "sources": [{"relative_path": "tests/test_auth.py"}, {"relative_path": ""}],
    }

    highlight = graph_highlight(result, "demo")

    assert highlight["node_ids"] == [
        AUTH_FILE,
        "demo::file::src/auth/helpers.py",
        "demo::file::src/api/routes.py",
        "demo::file::tests/test_auth.py",
        "demo::function::src/auth/helpers.py::assist",
    ]
    assert highlight["focus_node_id"] == AUTH_FILE


def test_highlight_is_capped_and_tolerates_an_empty_turn():
    many = {"navigation_nodes": [f"demo::file::f{i}.py" for i in range(MAX_HIGHLIGHT + 20)]}
    assert len(graph_highlight(many, "demo")["node_ids"]) == MAX_HIGHLIGHT
    assert graph_highlight({"navigation_nodes": ["not-an-id"]}, "demo") == {
        "node_ids": [],
        "focus_node_id": None,
    }


# ------------------------------------------------------ repository resolution


def test_single_repository_is_resolved_implicitly():
    assert resolve_repository(None, ["demo"]) == "demo"


def test_explicit_repository_is_honoured():
    assert resolve_repository("demo", ["demo", "other"]) == "demo"


def test_unknown_repository_is_rejected_with_available_names():
    with pytest.raises(UnknownRepositoryError, match="demo"):
        resolve_repository("nope", ["demo"])


def test_ambiguous_repository_is_rejected():
    with pytest.raises(UnknownRepositoryError, match="required"):
        resolve_repository(None, ["demo", "other"])


def test_no_repositories_gives_an_actionable_error():
    with pytest.raises(UnknownRepositoryError, match="Import one"):
        resolve_repository(None, [])


# ------------------------------------------------------------ URL validation


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/fastapi/fastapi",
        "https://github.com/fastapi/fastapi/",
        "https://github.com/fastapi/fastapi.git",
        "  https://www.github.com/encode/httpx  ",
    ],
)
def test_github_urls_are_accepted(url: str):
    clone_url, name = parse_github_url(url)
    assert clone_url.startswith("https://github.com/")
    assert clone_url.endswith(f"/{name}.git")
    assert name in ("fastapi", "httpx")


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/fastapi/fastapi",
        "https://gitlab.com/fastapi/fastapi",
        "ext::sh -c touch% /tmp/pwned",
        "file:///etc",
        "https://user:token@github.com/fastapi/fastapi",
        "https://github.com/fastapi",
        "https://github.com/fastapi/fastapi/tree/master",
        "https://github.com/fastapi/fastapi?x=1",
        "https://github.com/fastapi/-oProxyCommand=evil",
        "https://github.evil.com/fastapi/fastapi",
    ],
)
def test_unsafe_or_foreign_urls_are_refused(url: str):
    with pytest.raises(InvalidRepositoryUrl):
        parse_github_url(url)


# ------------------------------------------------------------ import service


class FailingEmbeddingService(FakeEmbeddingService):
    def generate_embeddings(self, texts):
        raise ConnectionError("Ollama is not running")


def make_indexing(tmp_path: Path, embedding=None) -> IndexingService:
    return IndexingService(
        batch_size=2,
        repositories_root=str(tmp_path),
        embedding_service=embedding or FakeEmbeddingService(),
        vector_store=VectorStore(persist_directory=str(tmp_path / "chroma")),
        graph_store=GraphStore(path=str(tmp_path / "graph.sqlite3")),
    )


def fake_clone(tmp_path: Path):
    def clone(url: str) -> dict:
        _, name = parse_github_url(url)
        root = tmp_path / "repositories" / name
        write(root, "app/main.py", "from app.util import Helper\n\ndef main():\n    return Helper().run()\n")
        write(root, "app/util.py", "class Helper:\n    def run(self):\n        return 42\n")
        return {"status": "success", "repository": name, "path": str(root)}

    return clone


URL = "https://github.com/acme/widgets"


def test_import_builds_graph_and_search(tmp_path: Path):
    indexed: list[str] = []
    imports = ImportService(
        make_indexing(tmp_path), clone=fake_clone(tmp_path),
        on_indexed=indexed.append, run_async=False,
    )

    job = imports.start(URL)

    assert job.repository == "widgets"
    assert job.state == STATE_READY
    assert job.graph_ready and job.search_ready
    assert indexed == ["widgets"]
    assert imports.status("widgets")["state"] == STATE_READY


def test_graph_is_built_before_embedding_starts(tmp_path: Path):
    indexing = make_indexing(tmp_path)
    fake_clone(tmp_path)(URL)
    seen: list[tuple[str, bool]] = []

    indexing.index_repository(
        tmp_path / "repositories" / "widgets",
        repository="widgets",
        progress=lambda phase, done, total: seen.append(
            (phase, indexing.graph_store.has_repository("widgets"))
        ),
    )

    phases = [phase for phase, _ in seen]
    assert phases[:2] == ["parsing", "graph"]
    first_embedding = phases.index("embedding")
    assert seen[first_embedding][1] is True
    assert phases[first_embedding:] == ["embedding"] * (len(phases) - first_embedding)


def test_failed_embedding_still_leaves_a_usable_graph(tmp_path: Path):
    imports = ImportService(
        make_indexing(tmp_path, FailingEmbeddingService()),
        clone=fake_clone(tmp_path), run_async=False,
    )

    job = imports.start(URL)

    assert job.state == STATE_ERROR
    assert job.graph_ready is True
    assert job.search_ready is False
    assert "graph is available" in job.message


def test_failed_clone_reports_the_git_message(tmp_path: Path):
    imports = ImportService(
        make_indexing(tmp_path),
        clone=lambda url: {"status": "error", "message": "Git error: not found"},
        run_async=False,
    )

    job = imports.start(URL)

    assert job.state == STATE_ERROR
    assert job.graph_ready is False
    assert job.message == "Git error: not found"


def test_invalid_url_starts_no_job(tmp_path: Path):
    imports = ImportService(make_indexing(tmp_path), clone=fake_clone(tmp_path), run_async=False)
    with pytest.raises(InvalidRepositoryUrl):
        imports.start("https://gitlab.com/acme/widgets")
    assert imports.jobs() == []


def test_importing_a_repository_already_in_flight_returns_that_job(tmp_path: Path):
    imports = ImportService(make_indexing(tmp_path), clone=fake_clone(tmp_path), run_async=False)
    running = ImportJob(repository="widgets", url=URL, state=STATE_CLONING)
    imports._jobs["widgets"] = running

    assert imports.start(URL) is running


def test_imports_index_one_at_a_time(tmp_path: Path):
    imports = ImportService(make_indexing(tmp_path), clone=fake_clone(tmp_path), run_async=True)
    imports._indexing_lock.acquire()  # another import is indexing
    try:
        job = imports.start(URL)
        deadline = time.monotonic() + 10
        while job.message != "Waiting for another import to finish." and time.monotonic() < deadline:
            time.sleep(0.01)
        assert job.message == "Waiting for another import to finish."
        assert job.state == STATE_CLONING
    finally:
        imports._indexing_lock.release()

    deadline = time.monotonic() + 30
    while not job.finished and time.monotonic() < deadline:
        time.sleep(0.02)
    assert job.state == STATE_READY
    assert job.message == ""


def test_reimporting_skips_work_already_done(tmp_path: Path):
    indexing = make_indexing(tmp_path)
    imports = ImportService(indexing, clone=fake_clone(tmp_path), run_async=False)
    imports.start(URL)
    calls = indexing.embedding_service.calls

    job = imports.start(URL)

    assert job.state == STATE_READY
    assert job.message == "Already imported."
    assert indexing.embedding_service.calls == calls


def test_status_of_a_graph_from_a_previous_session(tmp_path: Path):
    indexing = make_indexing(tmp_path)
    ImportService(indexing, clone=fake_clone(tmp_path), run_async=False).start(URL)

    fresh = ImportService(indexing, clone=fake_clone(tmp_path), run_async=False)

    assert fresh.status("widgets")["graph_ready"] is True
    assert fresh.status("missing") is None
