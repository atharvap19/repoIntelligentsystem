"""Graph service and HTTP layer (Phase 3, stages 3.4 and 3.9)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import graph as graph_api
from app.graph.builder import GraphBuilder
from app.graph.store import GraphStore
from app.services.graph_service import GraphNotFoundError, GraphService
from app.utils.redaction import redact

from .conftest import write

SERVICE_PY = '''\
from .helpers import assist


class AuthService:
    """Handles authentication."""

    def login(self, user):
        return assist(user)

    def logout(self):
        return None
'''


def parsed(path: str, content: str) -> dict:
    return {
        "repository": "demo",
        "relative_path": path,
        "file_name": path.split("/")[-1],
        "language": "python",
        "extension": ".py",
        "file_hash": "h",
        "content": content,
    }


FILES = [
    parsed("src/auth/service.py", SERVICE_PY),
    parsed("src/auth/helpers.py", "def assist(user):\n    return user\n"),
    parsed("src/auth/nested/deep.py", "def deep():\n    return 1\n"),
    parsed("src/api/routes.py", "from src.auth.service import AuthService\n"),
    parsed("tests/test_auth.py", "from src.auth.service import AuthService\n"),
]


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A real checkout on disk, so content reads are exercised for real."""
    root = tmp_path / "demo"
    for record in FILES:
        write(root, record["relative_path"], record["content"])
    return root


@pytest.fixture
def service(tmp_path: Path, checkout: Path) -> GraphService:
    store = GraphStore(path=str(tmp_path / "graph.sqlite3"))
    builder = GraphBuilder("demo")
    builder.build(FILES)
    store.replace_repository("demo", builder.graph.nodes, builder.graph.edges)
    store.set_meta("demo", "path", {"value": str(checkout)})

    store.replace_history(
        "demo",
        [
            {"sha": "a1", "author": "Ada", "authored_at": 1_000, "summary": "add auth", "is_merge": False},
            {"sha": "b2", "author": "Ada", "authored_at": 5_000, "summary": "add api", "is_merge": False},
            {"sha": "c3", "author": "Ada", "authored_at": 7_000, "summary": "add tests", "is_merge": False},
        ],
        [
            {"sha": "a1", "relative_path": "src/auth/service.py", "module": "src/auth",
             "change_type": "A", "authored_at": 1_000},
            {"sha": "a1", "relative_path": "src/auth/helpers.py", "module": "src/auth",
             "change_type": "A", "authored_at": 1_000},
            {"sha": "b2", "relative_path": "src/api/routes.py", "module": "src/api",
             "change_type": "A", "authored_at": 5_000},
            # Every module needs history, or it is treated as predating the
            # walked window and appears in every snapshot.
            {"sha": "c3", "relative_path": "tests/test_auth.py", "module": "tests",
             "change_type": "A", "authored_at": 7_000},
        ],
        [{"name": "v1.0", "kind": "tag", "sha": "b2", "created_at": 5_000}],
    )
    return GraphService(graph_store=store, repositories_root=str(tmp_path))


@pytest.fixture
def client(service: GraphService) -> TestClient:
    """An isolated app so the API never touches the real ./graph_db."""
    app = FastAPI()
    app.include_router(graph_api.router)
    original = graph_api.service
    graph_api.service = service
    try:
        yield TestClient(app)
    finally:
        graph_api.service = original


AUTH_FILE = "demo::file::src/auth/service.py"


# --------------------------------------------------------------- overview


def test_overview_returns_modules_not_files(service: GraphService):
    overview = service.overview("demo")
    kinds = {n["kind"] for n in overview["nodes"]}
    assert kinds <= {"module", "external"}
    assert {n["name"] for n in overview["nodes"] if n["kind"] == "module"} == {
        "src/auth", "src/api", "tests"
    }


def test_overview_rolls_up_file_counts(service: GraphService):
    overview = service.overview("demo")
    auth = next(n for n in overview["nodes"] if n["name"] == "src/auth")
    assert auth["data"]["file_count"] == 3
    assert auth["data"]["primary_language"] == "python"


def test_overview_includes_module_dependencies(service: GraphService):
    overview = service.overview("demo")
    pairs = {(e["source"].split("::")[-1], e["target"].split("::")[-1]) for e in overview["edges"]}
    assert ("src/api", "src/auth") in pairs


def test_overview_unknown_repository_raises(service: GraphService):
    with pytest.raises(GraphNotFoundError):
        service.overview("nope")


# ---------------------------------------------------------------- expand


def test_expand_repository_gives_modules(service: GraphService):
    root = service.store.roots("demo", "repository")[0]
    result = service.expand(root["id"])
    assert {n["kind"] for n in result["nodes"]} == {"module"}


def test_expand_module_gives_directories_and_files(service: GraphService):
    module = service.store.find_node("demo", "src/auth", "module")
    result = service.expand(module["id"])
    assert {n["kind"] for n in result["nodes"]} == {"directory", "file"}
    # Containers sort ahead of files so truncation keeps the useful entries.
    assert result["nodes"][0]["kind"] == "directory"


def test_expand_file_gives_symbols(service: GraphService):
    result = service.expand(AUTH_FILE)
    assert {n["name"] for n in result["nodes"]} >= {"AuthService"}


def test_expand_truncates_and_reports(service: GraphService):
    module = service.store.find_node("demo", "src/auth", "module")
    result = service.expand(module["id"], limit=1)
    assert len(result["nodes"]) == 1
    assert result["truncated"] is True
    assert result["total_children"] > 1


def test_expand_unknown_node_raises(service: GraphService):
    with pytest.raises(GraphNotFoundError):
        service.expand("demo::file::nope.py")


# ------------------------------------------------------------ dependencies


def test_dependencies_both_directions(service: GraphService):
    result = service.dependencies(AUTH_FILE)
    assert any(n["name"] == "helpers.py" for n in result["dependencies"])
    assert {n["name"] for n in result["dependents"]} == {"routes.py", "test_auth.py"}


def test_dependency_edges_are_returned_for_highlighting(service: GraphService):
    result = service.dependencies(AUTH_FILE)
    assert result["edges"]
    assert all("source" in e and "target" in e for e in result["edges"])


def test_node_detail_includes_symbols_and_history(service: GraphService):
    detail = service.node_detail(AUTH_FILE)
    assert any(s["name"] == "AuthService" for s in detail["symbols"])
    assert detail["history"][0]["sha"] == "a1"
    assert detail["dependents"]


# -------------------------------------------------------------- content


def test_file_content_is_read_from_disk(service: GraphService):
    content = service.file_content(AUTH_FILE)
    assert "class AuthService" in content["content"]
    assert content["language"] == "python"
    assert content["symbols"]


def test_file_content_rejects_non_file_nodes(service: GraphService):
    module = service.store.find_node("demo", "src/auth", "module")
    with pytest.raises(GraphNotFoundError):
        service.file_content(module["id"])


def test_file_content_redacts_secrets(service: GraphService, checkout: Path):
    write(checkout, "src/auth/config.py", 'API_KEY = "sk-abcdefghij1234567890"\n')
    builder = GraphBuilder("demo")
    builder.build(FILES + [parsed("src/auth/config.py", 'API_KEY = "sk-abcdefghij1234567890"\n')])
    service.store.replace_repository("demo", builder.graph.nodes, builder.graph.edges)

    content = service.file_content("demo::file::src/auth/config.py")
    assert "sk-abcdefghij1234567890" not in content["content"]
    assert content["redacted"] is True


def test_redaction_leaves_ordinary_code_alone():
    # Regression: an earlier rule rewrote `token = ctx_var.set(x)`.
    source = "token = _context_var.set(value)\naccess_token = await get_token()\n"
    assert redact(source).text == source


# --------------------------------------------------------------- timeline


def test_timeline_lanes_are_modules(service: GraphService):
    timeline = service.timeline("demo")
    assert {lane["module"] for lane in timeline["lanes"]} == {"src/auth", "src/api", "tests"}
    assert timeline["range"]["total"] == 3
    assert timeline["releases"][0]["name"] == "v1.0"


def test_snapshot_grows_over_time(service: GraphService):
    early = service.snapshot("demo", 2_000)
    late = service.snapshot("demo", 9_000)

    assert {n["name"] for n in early["nodes"]} == {"src/auth"}
    assert {n["name"] for n in late["nodes"]} >= {"src/auth", "src/api"}
    assert late["files_present"] >= early["files_present"]


def test_snapshot_counts_indexed_files_only(service: GraphService):
    # Counting raw history paths would mix denominators with file_count.
    snapshot = service.snapshot("demo", 9_000)
    auth = next(n for n in snapshot["nodes"] if n["name"] == "src/auth")
    assert auth["data"]["file_count_at"] <= auth["data"]["file_count"]


def test_snapshot_is_declared_approximate(service: GraphService):
    assert service.snapshot("demo", 9_000)["approximate"] is True


# ------------------------------------------------------------------ http


def test_http_repositories(client: TestClient):
    body = client.get("/graph/repositories").json()
    # The repository name must survive: `counts` has a `repository` key too.
    assert body["repositories"][0]["repository"] == "demo"
    assert body["repositories"][0]["counts"]["file"] == 5


def test_http_overview(client: TestClient):
    response = client.get("/graph/demo/overview")
    assert response.status_code == 200
    assert response.json()["repository"] == "demo"


def test_http_expand(client: TestClient):
    response = client.get("/graph/node/expand", params={"node_id": AUTH_FILE})
    assert response.status_code == 200
    assert response.json()["nodes"]


def test_http_dependencies(client: TestClient):
    body = client.get("/graph/node/dependencies", params={"node_id": AUTH_FILE}).json()
    assert body["dependents"]


def test_http_content(client: TestClient):
    body = client.get("/graph/node/content", params={"node_id": AUTH_FILE}).json()
    assert "AuthService" in body["content"]


def test_http_timeline_and_snapshot(client: TestClient):
    assert client.get("/graph/demo/timeline").status_code == 200
    assert client.get("/graph/demo/snapshot", params={"timestamp": 9000}).status_code == 200


def test_http_search(client: TestClient):
    body = client.get("/graph/demo/search", params={"q": "AuthService"}).json()
    assert any("AuthService" in r["name"] for r in body["results"])


def test_http_404s(client: TestClient):
    assert client.get("/graph/nope/overview").status_code == 404
    assert client.get("/graph/node", params={"node_id": "demo::file::nope.py"}).status_code == 404


def test_http_validation(client: TestClient):
    assert client.get("/graph/demo/search", params={"q": ""}).status_code == 422
    assert client.get("/graph/demo/snapshot").status_code == 422
