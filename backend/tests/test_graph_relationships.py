"""Relationship extraction: CONTAINS, IMPORTS, CALLS, INHERITS, IMPLEMENTS.

Every expected edge can be checked by reading the fixture source. Every edge
asserted *absent* is a case where drawing it would mean guessing.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.graph.builder import GraphBuilder, build_repository_graph
from app.graph.model import (
    EDGE_CALLS,
    EDGE_CONTAINS,
    EDGE_IMPLEMENTS,
    EDGE_IMPORTS,
    EDGE_INHERITS,
    Edge,
    split_node_id,
)
from app.graph.store import GraphStore
from app.rag.parser import RepositoryParser
from app.services.graph_service import GraphService

from .conftest import write


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


def build(files: dict[str, str]) -> GraphBuilder:
    builder = GraphBuilder("demo")
    builder.build([parsed(path, source) for path, source in files.items()])
    return builder


def key(identifier: str) -> str:
    return split_node_id(identifier)[2]


def relations(builder: GraphBuilder, kind: str) -> set[tuple[str, str]]:
    return {(key(e.source_id), key(e.target_id)) for e in builder.graph.edges if e.kind == kind}


def find(builder: GraphBuilder, kind: str, source: str, target: str) -> Edge:
    return next(
        e
        for e in builder.graph.edges
        if e.kind == kind and key(e.source_id) == source and key(e.target_id) == target
    )


# ----------------------------------------------------------- the fixture

USER_SERVICE = '''\
class UserService:
    def get_user(self):
        pass
'''

AUTH_SERVICE = '''\
from user_service import UserService


class AuthService:
    def login(self):
        user_service = UserService()
        return user_service.get_user()
'''

MAIN = '''\
from auth_service import AuthService


def main():
    auth = AuthService()
    auth.login()
'''

FIXTURE = {"user_service.py": USER_SERVICE, "auth_service.py": AUTH_SERVICE, "main.py": MAIN}


def test_fixture_imports():
    assert relations(build(FIXTURE), EDGE_IMPORTS) == {
        ("main.py", "auth_service.py"),
        ("auth_service.py", "user_service.py"),
    }


def test_fixture_containment():
    assert {
        ("user_service.py", "user_service.py::UserService"),
        ("user_service.py::UserService", "user_service.py::UserService.get_user"),
        ("auth_service.py", "auth_service.py::AuthService"),
        ("auth_service.py::AuthService", "auth_service.py::AuthService.login"),
        ("main.py", "main.py::main"),
    } <= relations(build(FIXTURE), EDGE_CONTAINS)


def test_fixture_calls():
    # Constructor calls land on the class: `UserService()` invokes the class.
    assert relations(build(FIXTURE), EDGE_CALLS) == {
        ("auth_service.py::AuthService.login", "user_service.py::UserService"),
        ("auth_service.py::AuthService.login", "user_service.py::UserService.get_user"),
        ("main.py::main", "auth_service.py::AuthService"),
        ("main.py::main", "auth_service.py::AuthService.login"),
    }


def test_fixture_edges_carry_file_and_line():
    builder = build(FIXTURE)

    call = find(builder, EDGE_CALLS, "auth_service.py::AuthService.login", "user_service.py::UserService.get_user")
    assert call.data["file"] == "auth_service.py"
    assert call.data["line"] == 7
    assert call.data["via"] == "local-instance"

    constructor = find(builder, EDGE_CALLS, "main.py::main", "auth_service.py::AuthService")
    assert (constructor.data["line"], constructor.data["via"]) == (5, "constructor")

    assert find(builder, EDGE_IMPORTS, "main.py", "auth_service.py").data == {
        "names": ["AuthService"], "line": 1, "file": "main.py"
    }
    assert find(
        builder, EDGE_CONTAINS, "auth_service.py::AuthService", "auth_service.py::AuthService.login"
    ).data == {"file": "auth_service.py", "line": 5}


def test_fixture_through_repository_parser(tmp_path: Path):
    root = tmp_path / "repo"
    for path, source in FIXTURE.items():
        write(root, path, source)

    graph, stats = build_repository_graph("demo", RepositoryParser().parse_repository(root))

    calls = {(key(e.source_id), key(e.target_id)) for e in graph.edges if e.kind == EDGE_CALLS}
    assert ("main.py::main", "auth_service.py::AuthService.login") in calls
    assert stats.calls_resolved == 4


# ------------------------------------------------ nothing is guessed at


def test_untyped_receivers_create_no_calls():
    builder = build(
        {
            "a.py": "class A:\n    def get_user(self):\n        pass\n",
            "b.py": "class B:\n    def get_user(self):\n        pass\n",
            "c.py": (
                "def load(repo):\n    return repo.get_user()\n\n\n"
                "def chained():\n    return make().get_user()\n\n\n"
                "def factory():\n    thing = make()\n    return thing.get_user()\n\n\n"
                "def builtin():\n    print(len([]))\n"
            ),
        }
    )
    assert relations(builder, EDGE_CALLS) == set()


def test_matching_name_without_import_is_not_a_call():
    builder = build(
        {
            "helpers.py": "def helper():\n    pass\n",
            "other.py": "def run():\n    helper()\n",
        }
    )
    assert relations(builder, EDGE_CALLS) == set()


def test_shadowed_names_are_not_resolved():
    builder = build(
        {
            "util.py": "def helper():\n    pass\n",
            "main.py": (
                "from util import helper\n\n\n"
                "def parameter(helper):\n    helper()\n\n\n"
                "def reassigned():\n    helper = make()\n    helper()\n\n\n"
                "def plain():\n    helper()\n"
            ),
        }
    )
    assert relations(builder, EDGE_CALLS) == {("main.py::plain", "util.py::helper")}


def test_conflicting_local_types_are_dropped():
    source = (
        "class A:\n    def go(self):\n        pass\n\n\n"
        "class B:\n    def go(self):\n        pass\n\n\n"
        "def pick(flag):\n    obj = A()\n    if flag:\n        obj = B()\n    obj.go()\n"
    )
    assert relations(build({"x.py": source}), EDGE_CALLS) == {
        ("x.py::pick", "x.py::A"),
        ("x.py::pick", "x.py::B"),
    }


def test_external_base_blocks_inherited_lookup():
    builder = build(
        {
            "base.py": "class Mixin:\n    def save(self):\n        pass\n",
            "model.py": (
                "from pydantic import BaseModel\n\nfrom base import Mixin\n\n\n"
                "class Model(BaseModel, Mixin):\n"
                "    def own(self):\n        pass\n\n"
                "    def run(self):\n        self.own()\n        self.save()\n"
            ),
        }
    )
    # `save` might be defined by BaseModel, which comes first in the MRO.
    assert relations(builder, EDGE_CALLS) == {("model.py::Model.run", "model.py::Model.own")}
    assert relations(builder, EDGE_INHERITS) == {("model.py::Model", "base.py::Mixin")}


# ------------------------------------------------------- call resolution

BASE = '''\
class Base:
    def __init__(self):
        pass

    def shared(self):
        return 1
'''

CHILD = '''\
from base import Base


class Child(Base):
    def __init__(self):
        super().__init__()

    def run(self):
        self.helper()
        return self.shared()

    def helper(self):
        pass

    @classmethod
    def create(cls):
        return cls()
'''


def test_self_inherited_and_super_calls():
    builder = build({"base.py": BASE, "child.py": CHILD})
    assert relations(builder, EDGE_CALLS) == {
        ("child.py::Child.__init__", "base.py::Base.__init__"),
        ("child.py::Child.run", "child.py::Child.helper"),
        ("child.py::Child.run", "base.py::Base.shared"),
        ("child.py::Child.create", "child.py::Child"),
    }
    assert relations(builder, EDGE_INHERITS) == {("child.py::Child", "base.py::Base")}
    assert find(builder, EDGE_CALLS, "child.py::Child.__init__", "base.py::Base.__init__").data["via"] == "super"


def test_module_class_and_reexported_calls():
    builder = build(
        {
            "pkg/__init__.py": "from .impl import Thing\n",
            "pkg/impl.py": (
                "class Thing:\n    def run(self):\n        pass\n\n"
                "    @staticmethod\n    def build():\n        return Thing()\n"
            ),
            "pkg/helpers.py": "def assist():\n    pass\n",
            "app.py": (
                "import pkg.helpers\nfrom pkg import helpers as h\nfrom pkg import Thing\n\n\n"
                "def go():\n    pkg.helpers.assist()\n    h.assist()\n    Thing.build()\n    Thing().run()\n"
            ),
        }
    )
    assert relations(builder, EDGE_CALLS) == {
        ("app.py::go", "pkg/helpers.py::assist"),
        ("app.py::go", "pkg/impl.py::Thing.build"),
        ("app.py::go", "pkg/impl.py::Thing"),
        ("app.py::go", "pkg/impl.py::Thing.run"),
        ("pkg/impl.py::Thing.build", "pkg/impl.py::Thing"),
    }
    assert find(builder, EDGE_CALLS, "app.py::go", "pkg/helpers.py::assist").data["count"] == 2
    assert relations(builder, EDGE_IMPORTS) == {
        ("app.py", "pkg/helpers.py"),
        ("app.py", "pkg/__init__.py"),
        ("pkg/__init__.py", "pkg/impl.py"),
    }


def test_typed_attributes_and_parameters():
    builder = build(
        {
            "repo.py": "class Repo:\n    def load(self):\n        pass\n",
            "svc.py": (
                "from typing import Optional\n\nfrom repo import Repo\n\n\n"
                "class Service:\n"
                "    def __init__(self):\n        self.repo = Repo()\n\n"
                "    def fetch(self):\n        return self.repo.load()\n\n\n"
                "def use(r: Repo):\n    r.load()\n\n\n"
                "def maybe(r: Optional[Repo] = None):\n    r.load()\n\n\n"
                "def quoted(r: 'Repo'):\n    r.load()\n"
            ),
        }
    )
    assert relations(builder, EDGE_CALLS) == {
        ("svc.py::Service.__init__", "repo.py::Repo"),
        ("svc.py::Service.fetch", "repo.py::Repo.load"),
        ("svc.py::use", "repo.py::Repo.load"),
        ("svc.py::maybe", "repo.py::Repo.load"),
        ("svc.py::quoted", "repo.py::Repo.load"),
    }


def test_star_import_respects_all():
    builder = build(
        {
            "lib/__init__.py": "from ._client import *\n",
            "lib/_client.py": (
                "__all__ = ['Client']\n\n\n"
                "class Client:\n    def send(self):\n        pass\n\n\n"
                "class Hidden:\n    pass\n"
            ),
            "use.py": "import lib\n\n\ndef go():\n    lib.Client().send()\n    lib.Hidden()\n",
        }
    )
    assert relations(builder, EDGE_CALLS) == {
        ("use.py::go", "lib/_client.py::Client"),
        ("use.py::go", "lib/_client.py::Client.send"),
    }


def test_nested_class_members():
    source = (
        "class Outer:\n"
        "    class Inner:\n        def ping(self):\n            pass\n\n"
        "    def use(self):\n        return self.Inner().ping()\n"
    )
    builder = build({"n.py": source})
    contains = relations(builder, EDGE_CONTAINS)
    assert ("n.py::Outer", "n.py::Outer.Inner") in contains
    assert ("n.py::Outer.Inner", "n.py::Outer.Inner.ping") in contains
    assert relations(builder, EDGE_CALLS) == {
        ("n.py::Outer.use", "n.py::Outer.Inner"),
        ("n.py::Outer.use", "n.py::Outer.Inner.ping"),
    }


# ----------------------------------------------------- inheritance kinds


def test_protocol_subclass_implements():
    source = (
        "from abc import ABC\nfrom typing import Protocol\n\n\n"
        "class Store(Protocol):\n    def get(self):\n        ...\n\n\n"
        "class Base(ABC):\n    pass\n\n\n"
        "class MemoryStore(Store, Base):\n    def get(self):\n        return 1\n"
    )
    builder = build({"stores.py": source})
    assert relations(builder, EDGE_IMPLEMENTS) == {("stores.py::MemoryStore", "stores.py::Store")}
    # Subclassing an ABC is ordinary inheritance.
    assert relations(builder, EDGE_INHERITS) == {("stores.py::MemoryStore", "stores.py::Base")}


# ---------------------------------------------------- import resolution


def test_import_resolution():
    builder = build(
        {
            "pkg/__init__.py": "VERSION = 1\n",
            "pkg/parser.py": "def parse():\n    pass\n",
            "pkg/json.py": "LOADED = True\n",
            "pkg/local.py": "from . import parser\n",
            "ns/tools/cli.py": "def run():\n    pass\n",
            "client.py": "import json\nfrom pkg import parser\nfrom ns.tools import cli\n",
        }
    )
    # `from pkg import parser` names the submodule, not the package; `import
    # json` is the standard library, not pkg/json.py; `ns` has no __init__.py.
    assert relations(builder, EDGE_IMPORTS) == {
        ("client.py", "pkg/parser.py"),
        ("client.py", "ns/tools/cli.py"),
        ("pkg/local.py", "pkg/parser.py"),
    }
    assert builder.stats.external_imports == 1


# -------------------------------------------------------- determinism


def test_build_is_deterministic():
    files = {**FIXTURE, "base.py": BASE, "child.py": CHILD}

    def snapshot(order: list[str]) -> list[tuple[str, str]]:
        builder = GraphBuilder("demo")
        builder.build([parsed(path, files[path]) for path in order])
        return sorted((e.id, json.dumps(e.data, sort_keys=True)) for e in builder.graph.edges)

    order = sorted(files)
    assert snapshot(order) == snapshot(list(reversed(order)))


# --------------------------------------------------------------- api

SVC = '''\
class Repo:
    def load(self):
        return 1


class Service(Repo):
    def run(self):
        return self.load() + self.twice()

    def twice(self):
        return 2


def main():
    return Service().run()
'''


def test_expand_returns_symbol_relationships(tmp_path: Path):
    builder = build({"svc.py": SVC})
    store = GraphStore(path=str(tmp_path / "graph.sqlite3"))
    store.replace_repository("demo", builder.graph.nodes, builder.graph.edges)
    service = GraphService(graph_store=store)

    def drawn(view: dict) -> set[tuple[str, str, str]]:
        return {(e["kind"], key(e["source"]), key(e["target"])) for e in view["edges"]}

    file_view = drawn(service.expand("demo::file::svc.py"))
    assert ("INHERITS", "svc.py::Service", "svc.py::Repo") in file_view
    assert ("CALLS", "svc.py::main", "svc.py::Service") in file_view

    class_view = drawn(service.expand("demo::class::svc.py::Service"))
    assert class_view == {("CALLS", "svc.py::Service.run", "svc.py::Service.twice")}

    dependencies = service.dependencies("demo::method::svc.py::Service.run")
    assert {n["name"] for n in dependencies["dependencies"]} == {"Repo.load", "Service.twice"}
    assert {n["name"] for n in dependencies["dependents"]} == {"main"}
