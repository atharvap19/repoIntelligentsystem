"""Phase 4: hybrid context, Explorer awareness, navigation and insights.

The fixture repository is a miniature layered application — routes import
services, services import a repository module, which imports the database —
so flow, hotspot and dependency questions have a real shape to find rather
than a shape asserted by the test.

The LLM is faked throughout. What is under test is which evidence reaches the
prompt and what the backend decides about it, never generation quality.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.agent.context import (
    ContextBuilder,
    ConversationTurn,
    ExplorerContext,
    RepositoryContext,
)
from app.agent.graph import RepositoryAgent
from app.agent.intents import (
    INTENT_CONCEPT,
    INTENT_DEPENDENCY,
    INTENT_FLOW,
    INTENT_HOTSPOT,
    INTENT_READONLY,
    classify,
    extract_concepts,
    extract_mentions,
)
from app.agent.navigation import (
    FOCUS_NEIGHBOURHOOD,
    TARGET_ARCHITECTURE,
    TARGET_FILE,
    TARGET_TIMELINE,
    TARGET_TYPES,
    NavigationPlanner,
)
from app.api import graph as graph_api
from app.graph.builder import GraphBuilder
from app.graph.insights import InsightAnalyzer, render_flow, render_hotspots
from app.graph.store import GraphStore
from app.rag.graph_retriever import GraphRetriever, render_graph_context
from app.services.graph_service import GraphService
from app.services.llm_service import LLMService
from app.tools.architecture import ArchitectureTool
from app.tools.file_tool import FileTool

from .conftest import write

# ------------------------------------------------------------------ fixture

DATABASE_PY = "class Database:\n    def query(self, sql):\n        return []\n"

REPOSITORY_PY = '''\
from src.db.database import Database


class UserRepository:
    """Reads and writes users."""

    def __init__(self):
        self.db = Database()

    def find(self, user_id):
        return self.db.query("select 1")
'''

SERVICE_PY = '''\
from src.data.repository import UserRepository


class AuthService:
    """Handles authentication."""

    def login(self, user):
        return UserRepository().find(user)
'''

ROUTES_PY = '''\
from src.auth.service import AuthService


def login_route(request):
    return AuthService().login(request.user)
'''

MAIN_PY = "from src.api.routes import login_route\n\n\ndef main():\n    return login_route\n"

FILES = [
    {"relative_path": "src/db/database.py", "content": DATABASE_PY},
    {"relative_path": "src/data/repository.py", "content": REPOSITORY_PY},
    {"relative_path": "src/auth/service.py", "content": SERVICE_PY},
    {"relative_path": "src/api/routes.py", "content": ROUTES_PY},
    {"relative_path": "src/app/main.py", "content": MAIN_PY},
    {
        "relative_path": "src/api/admin.py",
        "content": "from src.auth.service import AuthService\n",
    },
    {
        "relative_path": "tests/test_auth.py",
        "content": "from src.auth.service import AuthService\n",
    },
]


def parsed(record: dict) -> dict:
    path = record["relative_path"]
    return {
        "repository": "demo",
        "relative_path": path,
        "file_name": path.split("/")[-1],
        "language": "python",
        "extension": ".py",
        "file_hash": "h",
        "content": record["content"],
    }


PARSED = [parsed(record) for record in FILES]

COMMITS = [
    {"sha": "aaaa1111", "author": "Ada", "authored_at": 1_000, "summary": "add database", "is_merge": False},
    {"sha": "bbbb2222", "author": "Grace", "authored_at": 2_000, "summary": "add repository", "is_merge": False},
    {"sha": "cccc3333", "author": "Ada", "authored_at": 3_000, "summary": "add auth service", "is_merge": False},
    {"sha": "dddd4444", "author": "Alan", "authored_at": 4_000, "summary": "wire routes", "is_merge": False},
    {"sha": "eeee5555", "author": "Ada", "authored_at": 5_000, "summary": "harden auth", "is_merge": False},
]

COMMIT_FILES = [
    {"sha": "aaaa1111", "relative_path": "src/db/database.py", "module": "src/db", "change_type": "A", "authored_at": 1_000},
    {"sha": "bbbb2222", "relative_path": "src/data/repository.py", "module": "src/data", "change_type": "A", "authored_at": 2_000},
    {"sha": "cccc3333", "relative_path": "src/auth/service.py", "module": "src/auth", "change_type": "A", "authored_at": 3_000},
    {"sha": "dddd4444", "relative_path": "src/api/routes.py", "module": "src/api", "change_type": "A", "authored_at": 4_000},
    {"sha": "dddd4444", "relative_path": "src/auth/service.py", "module": "src/auth", "change_type": "M", "authored_at": 4_000},
    {"sha": "eeee5555", "relative_path": "src/auth/service.py", "module": "src/auth", "change_type": "M", "authored_at": 5_000},
]


@pytest.fixture
def store(tmp_path: Path) -> GraphStore:
    checkout = tmp_path / "demo"
    for record in PARSED:
        write(checkout, record["relative_path"], record["content"])

    store = GraphStore(path=str(tmp_path / "graph.sqlite3"))
    builder = GraphBuilder("demo")
    builder.build(PARSED)
    store.replace_repository("demo", builder.graph.nodes, builder.graph.edges)
    store.set_meta("demo", "path", {"value": str(checkout)})
    store.replace_history(
        "demo",
        COMMITS,
        COMMIT_FILES,
        [{"name": "v1.0", "kind": "tag", "sha": "cccc3333", "created_at": 3_000}],
    )
    return store


@pytest.fixture
def service(store: GraphStore, tmp_path: Path) -> GraphService:
    return GraphService(graph_store=store, repositories_root=str(tmp_path))


@pytest.fixture
def retriever(service: GraphService) -> GraphRetriever:
    return GraphRetriever(service)


NODE = "demo::file::{}".format


class FakeSearchTool:
    """Returns one chunk per known file, scored so ordering is testable."""

    def __init__(self):
        self.calls: list[dict] = []

    def run(self, query, repository, top_k=5, language=None, chunk_type=None):
        self.calls.append({"query": query, "repository": repository, "top_k": top_k})
        return [
            {
                "relative_path": record["relative_path"],
                "file_name": record["relative_path"].split("/")[-1],
                "repository": repository,
                "symbol": "",
                "chunk_type": "module",
                "chunk_index": 0,
                "start_line": 1,
                "end_line": 10,
                "language": "python",
                "distance": 0.2,
                # Descending, so any reordering is attributable to the context
                # builder rather than to retrieval.
                "score": 1.0 - index * 0.1,
                "vector_score": 1.0 - index * 0.1,
                "keyword_score": 0.0,
                "content": record["content"],
            }
            for index, record in enumerate(FILES[:top_k])
        ]

    def invalidate(self, repository=None):
        pass


@pytest.fixture
def agent(service: GraphService) -> RepositoryAgent:
    return RepositoryAgent(
        graph_service=service,
        llm_service=LLMService(llm=FakeListChatModel(responses=["A grounded answer."] * 60)),
        search_tool=FakeSearchTool(),
        file_tool=FileTool(service),
        architecture_tool=ArchitectureTool(service),
    )


def ask(agent: RepositoryAgent, question: str, conversation_id: str = "t", **kwargs) -> dict:
    return agent.ask(question, repository="demo", conversation_id=conversation_id, **kwargs)


@pytest.fixture
def client(service: GraphService) -> TestClient:
    app = FastAPI()
    app.include_router(graph_api.router)
    graph_api.service = service
    return TestClient(app)


# ==================================================== 1. graph retrieval


def test_a_named_symbol_seeds_the_neighbourhood(retriever: GraphRetriever):
    context = retriever.retrieve("demo", mentions=["AuthService"])
    assert context.seeds
    assert context.by_id(context.seeds[0]).name in ("AuthService", "service.py")


def test_a_named_file_resolves_to_the_file_not_a_symbol(retriever: GraphRetriever):
    context = retriever.retrieve("demo", mentions=["service.py"])
    assert context.by_id(context.seeds[0]).kind == "file"


def test_both_directions_are_walked(retriever: GraphRetriever):
    # service.py imports repository.py and is imported by routes.py; an answer
    # about it needs the shape, not one arrow.
    names = {hit.name for hit in retriever.retrieve("demo", mentions=["service.py"]).hits}
    assert "repository.py" in names
    assert "routes.py" in names


def test_reasons_explain_why_each_node_is_present(retriever: GraphRetriever):
    context = retriever.retrieve("demo", mentions=["service.py"])
    reasons = {hit.name: hit.reason for hit in context.hits}
    assert reasons["service.py"] == "mention"
    assert reasons["repository.py"] == "imports"
    assert reasons["routes.py"] == "imported-by"


def test_the_seed_gets_its_containers(retriever: GraphRetriever):
    # A bare symbol node has no address without its file and module.
    kinds = {hit.kind for hit in retriever.retrieve("demo", mentions=["service.py"]).hits}
    assert "module" in kinds


def test_edges_are_only_kept_when_both_ends_survived(retriever: GraphRetriever):
    context = retriever.retrieve("demo", mentions=["service.py"])
    ids = context.node_ids
    for edge in context.edges:
        assert edge["source"] in ids
        assert edge["target"] in ids


def test_the_node_budget_is_enforced(service: GraphService):
    small = GraphRetriever(service, node_budget=3)
    context = small.retrieve("demo", mentions=["service.py"])
    assert len(context.hits) <= 3


def test_exceeding_the_budget_is_reported_not_hidden(service: GraphService):
    small = GraphRetriever(service, node_budget=2, hops=2)
    assert small.retrieve("demo", mentions=["service.py"]).withheld > 0


def test_explorer_selection_seeds_when_nothing_is_named(retriever: GraphRetriever):
    context = retriever.retrieve(
        "demo", mentions=[], explorer_node_ids=[NODE("src/auth/service.py")]
    )
    assert context.by_id(context.seeds[0]).name == "service.py"
    assert context.by_id(context.seeds[0]).reason == "explorer"


def test_retrieved_paths_seed_when_nothing_else_does(retriever: GraphRetriever):
    context = retriever.retrieve("demo", chunk_paths=["src/auth/service.py"])
    assert context.by_id(context.seeds[0]).reason == "retrieval"


def test_a_name_that_matches_nothing_yields_an_empty_context(retriever: GraphRetriever):
    assert not retriever.retrieve("demo", mentions=["nonexistent_xyz"])


def test_an_unindexed_repository_is_survivable(retriever: GraphRetriever):
    # Retrieval falling back to text-only beats a 500.
    assert not retriever.retrieve("no-such-repo", mentions=["anything"])


# -- regressions measured against the real FastAPI graph ---------------------
#
# Each of these encodes a failure observed on the 6,363-node checkout, not a
# hypothetical. They are the tests that make the brief's own worked example
# ("How does APIRouter connect to FastAPI?") actually work.


def test_a_symbol_seed_reaches_the_import_edges_of_its_file(retriever: GraphRetriever):
    # Import edges are recorded between files, never between symbols. Seeding
    # on the class `AuthService` and walking IMPORTS from it finds nothing —
    # on the real graph this returned 5 nodes and 0 edges for the brief's own
    # example. The containing file has to join the traversal.
    context = retriever.retrieve("demo", mentions=["AuthService"])
    assert context.edges
    names = {hit.name for hit in context.hits}
    assert "repository.py" in names   # what service.py imports
    assert "routes.py" in names       # what imports service.py


def test_the_defining_file_is_named_in_the_rendered_subject(retriever: GraphRetriever):
    text = render_graph_context(retriever.retrieve("demo", mentions=["AuthService"]))
    assert "src/auth/service.py:" in text
    assert "depends on ->" in text


def test_an_exact_name_beats_a_path_substring_match(store: GraphStore, service: GraphService):
    # Every path under `fastapi/` contains the string "fastapi", so a LIKE
    # search for `FastAPI` matches all 956 files and the shortest-named one
    # wins the tie-break. The class actually called FastAPI was never reached.
    hits = service.search("demo", "auth", limit=40)
    assert len(hits) > 1  # 'auth' is a substring of several paths here

    context = GraphRetriever(service).retrieve("demo", mentions=["AuthService"])
    seed = context.by_id(context.seeds[0])
    assert seed.name == "AuthService"


def test_a_hub_node_cannot_flood_the_neighbourhood(service: GraphService):
    # `fastapi/__init__.py` has 580 importers. Expanding it unranked fills the
    # entire budget with files named test_tutorial008d.py and the answer never
    # reaches applications.py.
    small = GraphRetriever(service, node_budget=24)
    context = small.retrieve("demo", mentions=["service.py"])
    per_source: dict[str, int] = {}
    for hit in context.hits:
        if hit.reason in ("imports", "imported-by"):
            per_source[hit.via] = per_source.get(hit.via, 0) + 1
    assert all(count <= 12 for count in per_source.values())


def test_implementation_outranks_tests_in_the_neighbourhood(retriever: GraphRetriever):
    # service.py is imported by routes.py, admin.py and tests/test_auth.py.
    # With a budget too tight for all three, the test is what gets dropped —
    # FastAPI's library is under 5% of its indexed files, so without this the
    # neighbourhood of any library symbol is almost entirely its own tests.
    tight = GraphRetriever(retriever.graph, node_budget=5)
    names = [hit.name for hit in tight.retrieve("demo", mentions=["service.py"]).hits]
    assert "routes.py" in names
    assert "admin.py" in names
    assert "test_auth.py" not in names


def test_rendered_context_states_relationships(retriever: GraphRetriever):
    text = render_graph_context(retriever.retrieve("demo", mentions=["service.py"]))
    assert "service.py" in text
    assert "depends on ->" in text
    assert "used by <-" in text


def test_rendering_an_empty_context_is_empty(retriever: GraphRetriever):
    assert render_graph_context(retriever.retrieve("demo", mentions=["nope_xyz"])) == ""


# ==================================================== 2. context building


def chunk(path: str, score: float, index: int = 0, **extra) -> dict:
    base = {
        "relative_path": path,
        "chunk_index": index,
        "content": f"# {path}",
        "score": score,
        "vector_score": score,
        "keyword_score": 0.0,
        "symbol": "",
        "chunk_type": "function",
    }
    base.update(extra)
    return base


def test_duplicate_chunks_collapse():
    built = ContextBuilder().build(
        query="q", repository="demo",
        chunks=[chunk("a.py", 0.5), chunk("a.py", 0.9)],
    )
    assert len(built.code) == 1


def test_deduplication_keeps_the_best_evidence_from_each_appearance():
    built = ContextBuilder().build(
        query="q", repository="demo",
        chunks=[
            chunk("a.py", 0.5, vector_score=0.5, keyword_score=0.0),
            chunk("a.py", 0.3, vector_score=0.0, keyword_score=0.8),
        ],
    )
    # Found twice by different strategies; neither channel is discarded.
    assert built.code[0]["vector_score"] == 0.5
    assert built.code[0]["keyword_score"] == 0.8


def test_different_chunks_of_one_file_are_not_collapsed():
    built = ContextBuilder().build(
        query="q", repository="demo",
        chunks=[chunk("a.py", 0.5, index=0), chunk("a.py", 0.4, index=1)],
    )
    assert len(built.code) == 2


def test_the_graph_promotes_a_chunk_retrieval_ranked_lower(retriever: GraphRetriever):
    graph = retriever.retrieve("demo", mentions=["service.py"])
    built = ContextBuilder().build(
        query="explain service.py", repository="demo",
        chunks=[chunk("unrelated.py", 0.9), chunk("src/auth/service.py", 0.6)],
        graph=graph,
    )
    # This is the fusion the brief asks for: structural evidence reorders a
    # near-tie that text scores alone would have got wrong.
    assert built.code[0]["relative_path"] == "src/auth/service.py"


def test_promotion_is_bounded_and_cannot_overturn_a_decisive_match(retriever: GraphRetriever):
    graph = retriever.retrieve("demo", mentions=["service.py"])
    built = ContextBuilder().build(
        query="q", repository="demo",
        chunks=[chunk("unrelated.py", 0.95), chunk("src/auth/service.py", 0.10)],
        graph=graph,
    )
    assert built.code[0]["relative_path"] == "unrelated.py"


def test_graph_linked_chunks_are_marked_so_the_prompt_can_say_why(retriever: GraphRetriever):
    built = ContextBuilder().build(
        query="q", repository="demo",
        chunks=[chunk("src/auth/service.py", 0.5)],
        graph=retriever.retrieve("demo", mentions=["service.py"]),
    )
    assert built.code[0]["graph_linked"] is True
    assert "reached via the repository graph" in built.render()


def test_the_code_budget_is_enforced():
    built = ContextBuilder(code_budget=3).build(
        query="q", repository="demo",
        chunks=[chunk(f"f{i}.py", 1.0 - i / 10) for i in range(10)],
    )
    assert len(built.code) == 3


def test_conversation_is_trimmed_to_the_recent_turns():
    built = ContextBuilder(history_turns=2).build(
        query="q", repository="demo",
        conversation=[ConversationTurn(question=f"q{i}", answer="a") for i in range(6)],
    )
    assert [t.question for t in built.conversation] == ["q4", "q5"]


def test_a_long_previous_answer_is_summarised_not_replayed_whole():
    built = ContextBuilder().build(
        query="q", repository="demo",
        conversation=[ConversationTurn(question="q", answer="x" * 5_000)],
    )
    assert len(built.render()) < 2_000


def test_an_oversized_excerpt_is_truncated_visibly():
    built = ContextBuilder().build(
        query="q", repository="demo", chunks=[chunk("a.py", 1.0, content="y" * 10_000)],
    )
    assert "(excerpt truncated)" in built.render()


def test_render_labels_every_block(retriever: GraphRetriever):
    built = ContextBuilder().build(
        query="q", repository="demo",
        chunks=[chunk("a.py", 1.0)],
        graph=retriever.retrieve("demo", mentions=["service.py"]),
        explorer=ExplorerContext(file="service.py"),
        conversation=[ConversationTurn(question="earlier", answer="answer")],
        facts=["a fact"],
        history="a commit",
    )
    text = built.render()
    for heading in (
        "CURRENT EXPLORER CONTEXT",
        "CONVERSATION SO FAR",
        "REPOSITORY GRAPH",
        "REPOSITORY FACTS",
        "REPOSITORY HISTORY",
        "RELEVANT CODE",
    ):
        assert heading in text


def test_empty_sources_render_nothing():
    assert ContextBuilder().build(query="q", repository="demo").render() == ""


def test_an_empty_context_is_reported_as_empty():
    assert ContextBuilder().build(query="q", repository="demo").is_empty()


def test_a_graph_only_context_is_not_empty(retriever: GraphRetriever):
    built = ContextBuilder().build(
        query="q", repository="demo",
        graph=retriever.retrieve("demo", mentions=["service.py"]),
    )
    assert not built.is_empty()


def test_the_two_retrieval_channels_stay_distinguishable():
    built = ContextBuilder().build(
        query="q", repository="demo",
        chunks=[
            chunk("a.py", 0.5, vector_score=0.5, keyword_score=0.0),
            chunk("b.py", 0.4, vector_score=0.0, keyword_score=0.4),
        ],
    )
    assert [c["relative_path"] for c in built.semantic_results] == ["a.py"]
    assert [c["relative_path"] for c in built.lexical_results] == ["b.py"]


def test_stats_are_reported_for_diagnosis():
    built = ContextBuilder(code_budget=2).build(
        query="q", repository="demo",
        chunks=[chunk("a.py", 0.5), chunk("a.py", 0.5), chunk("b.py", 0.4), chunk("c.py", 0.3)],
    )
    assert built.stats["chunks_in"] == 4
    assert built.stats["chunks_deduped"] == 3
    assert built.stats["chunks_used"] == 2


# ---------------------------------------------------------- explorer context


def test_explorer_context_renders_the_selection():
    text = ExplorerContext(
        repository="demo", module="src/auth", file="service.py", symbol="AuthService"
    ).render()
    assert "service.py" in text
    assert "AuthService" in text


def test_explorer_context_tells_the_model_to_resolve_pronouns():
    assert '"this"' in ExplorerContext(file="service.py").render()


def test_an_empty_explorer_context_renders_nothing():
    assert ExplorerContext().render() == ""


def test_explorer_seed_ids_are_ordered_most_precise_first():
    explorer = ExplorerContext(node_id="a", focus_node_ids=["b", "a"])
    assert explorer.seed_ids() == ["a", "b"]


# ======================================================== 3. navigation


def test_every_declared_target_type_is_distinct():
    assert len(set(TARGET_TYPES)) == len(TARGET_TYPES)


def test_architecture_questions_navigate_to_the_architecture(agent: RepositoryAgent):
    navigation = ask(agent, "Explain the architecture.")["navigation"]
    assert navigation["available"] is True
    assert navigation["target_type"] == TARGET_ARCHITECTURE


def test_a_dependency_question_opens_the_neighbourhood(agent: RepositoryAgent):
    navigation = ask(agent, "What depends on service.py?")["navigation"]
    assert navigation["available"] is True
    assert navigation["focus_mode"] == FOCUS_NEIGHBOURHOOD
    assert navigation["related_node_ids"]


def test_navigation_carries_the_exact_node_id(agent: RepositoryAgent):
    navigation = ask(agent, "Tell me about service.py")["navigation"]
    assert navigation["node_id"] == NODE("src/auth/service.py")


def test_navigation_names_a_real_place(agent: RepositoryAgent):
    assert "service.py" in ask(agent, "Tell me about service.py")["navigation"]["label"]


def test_a_history_question_navigates_the_timeline(agent: RepositoryAgent):
    navigation = ask(agent, "What changed recently?")["navigation"]
    assert navigation["target_type"] == TARGET_TIMELINE


def test_a_locating_question_points_at_the_best_file(agent: RepositoryAgent):
    navigation = ask(agent, "Where is authentication implemented?")["navigation"]
    assert navigation["available"] is True
    assert navigation["target_type"] in (TARGET_FILE, "class", "symbol", "function")


def test_navigation_is_decided_without_reading_the_answer(agent: RepositoryAgent):
    # Part 6: the frontend must never parse model output to navigate, so the
    # decision must not depend on what the model said.
    planner = NavigationPlanner()
    context = RepositoryContext(query="q", repository="demo", intent="architecture_analysis")
    assert planner.decide(context).available is True


def test_a_question_with_no_visual_target_offers_none():
    # Part 5: "How many Python files are there?" is answered, not explored.
    planner = NavigationPlanner()
    context = RepositoryContext(query="how many files?", repository="demo", intent="code_explanation")
    decision = planner.decide(context)
    assert decision.available is False
    assert decision.reason


# ======================================================== 4. query understanding


@pytest.mark.parametrize(
    "question,intent",
    [
        ("How does a request flow through this repository?", INTENT_FLOW),
        ("Trace the request path.", INTENT_FLOW),
        ("Show me the database layer", INTENT_CONCEPT),
        ("Show me everything related to authentication", INTENT_CONCEPT),
        ("Which modules changed the most?", INTENT_HOTSPOT),
        ("What are the structural hotspots?", INTENT_HOTSPOT),
        ("Rewrite indexing.py", INTENT_READONLY),
        ("Refactor the auth service", INTENT_READONLY),
    ],
)
def test_phase4_question_shapes_route(question: str, intent: str):
    assert classify(question).intent == intent


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Explain my indexing.py", "indexing.py"),
        ("Explain the indexing file", "indexing"),
        ("What does the auth module do?", "auth"),
        ("Explain module app.rag.parser", "app.rag.parser"),
        ("What does the function called build_context do?", "build_context"),
        # A bare snake_case name, which is how Python functions get typed into
        # a question. Without this it resolves nothing and the turn falls back
        # to a blind vector search.
        ("What does create_access_token do?", "create_access_token"),
        ("Where is get_current_user defined?", "get_current_user"),
    ],
)
def test_names_are_extracted_with_or_without_an_extension(question: str, expected: str):
    assert expected in extract_mentions(question)


@pytest.mark.parametrize(
    "question",
    ["How does the login flow work?", "Explain the whole thing", "What is this doing?"],
)
def test_prose_is_not_mistaken_for_an_identifier(question: str):
    # The underscore is what makes _IDENTIFIER_MENTION safe; ordinary English
    # has none, so no plain-word question should produce a mention from it.
    assert all("_" not in m for m in extract_mentions(question))


@pytest.mark.parametrize(
    "question",
    [
        "What is this function doing?",
        "Why does this file depend on parser.py?",
        "How has this module evolved?",
    ],
)
def test_verbs_after_a_structural_noun_are_not_names(question: str):
    # "function doing", "file depend", "module evolved" are English, not
    # identifiers; treating them as names sends focus to a node that is not there.
    for value in extract_mentions(question):
        assert value.lower() not in ("doing", "depend", "evolved")


@pytest.mark.parametrize("word", ["Python", "JavaScript", "SQL", "React"])
def test_language_names_are_not_symbols(word: str):
    # "How many Python files are there?" otherwise resolves focus onto a
    # directory called `python_types` and then offers it as a place to
    # explore — the false positive Part 5 warns against.
    assert word not in extract_mentions(f"How many {word} files are in this repository?")


def test_a_counting_question_offers_no_exploration(agent: RepositoryAgent):
    assert (
        ask(agent, "How many Python files are in this repository?")["navigation"]["available"]
        is False
    )


def test_concepts_are_extracted_from_themes():
    assert "database" in extract_concepts("Show me the database layer")
    assert "authentication" in extract_concepts("everything related to authentication")


def test_structural_nouns_are_not_concepts():
    assert "class" not in extract_concepts("Tell me about the class AuthService")


# ======================================================== 5. insights


def test_the_most_imported_file_is_a_structural_hotspot(store: GraphStore):
    hotspots = InsightAnalyzer(store).hotspots("demo")
    structural = [h for h in hotspots if h.category == "structural"]
    assert structural
    assert structural[0].name == "service.py"


def test_a_hotspot_states_the_count_behind_it(store: GraphStore):
    structural = [h for h in InsightAnalyzer(store).hotspots("demo") if h.category == "structural"]
    # Part 13: explain *why*, from a real number, never an assertion.
    assert str(structural[0].value) in structural[0].reason
    assert structural[0].evidence


def test_churn_is_measured_from_real_commits(store: GraphStore):
    churn = [h for h in InsightAnalyzer(store).hotspots("demo") if h.category == "churn"]
    assert churn[0].path == "src/auth/service.py"
    assert churn[0].value == 3


def test_churn_ignores_paths_that_are_not_indexed(store: GraphStore):
    store.replace_history(
        "demo",
        COMMITS,
        COMMIT_FILES + [
            {"sha": s, "relative_path": "README.md", "module": "(root)",
             "change_type": "M", "authored_at": t}
            for s, t in (("aaaa1111", 1_000), ("bbbb2222", 2_000), ("cccc3333", 3_000),
                         ("dddd4444", 4_000), ("eeee5555", 5_000))
        ],
        [],
    )
    churn = [h for h in InsightAnalyzer(store).hotspots("demo") if h.category == "churn"]
    # README.md changes most, but it is not in the index — reporting it would
    # answer a question about a file the user cannot open here.
    assert all(h.path != "README.md" for h in churn)


def test_hotspots_render_with_their_headings(store: GraphStore):
    text = render_hotspots(InsightAnalyzer(store).hotspots("demo"))
    assert "Most depended-on" in text
    assert "import edges" in text


def test_a_repository_with_no_hotspots_reports_none(tmp_path: Path):
    empty = GraphStore(path=str(tmp_path / "empty.sqlite3"))
    builder = GraphBuilder("solo")
    builder.build([{"repository": "solo", "relative_path": "a.py", "file_name": "a.py",
                    "language": "python", "extension": ".py", "content": "x = 1\n"}])
    empty.replace_repository("solo", builder.graph.nodes, builder.graph.edges)
    assert not [h for h in InsightAnalyzer(empty).hotspots("solo") if h.category == "structural"]


def test_a_flow_starts_at_a_conventional_entry_point(store: GraphStore):
    flow = InsightAnalyzer(store).flow("demo")
    assert flow
    assert flow.steps[0].nodes[0]["name"] == "main.py"


def test_a_flow_follows_real_imports(store: GraphStore):
    flow = InsightAnalyzer(store).flow("demo")
    reached = [n["name"] for step in flow.steps for n in step.nodes]
    assert reached[:3] == ["main.py", "routes.py", "service.py"]


def test_a_flow_is_depth_limited(store: GraphStore):
    assert len(InsightAnalyzer(store).flow("demo").steps) <= 4


def test_a_flow_can_start_from_a_chosen_node(store: GraphStore):
    flow = InsightAnalyzer(store).flow("demo", NODE("src/auth/service.py"))
    assert flow.steps[0].nodes[0]["name"] == "service.py"


def test_a_rendered_flow_marks_itself_as_static(store: GraphStore):
    text = render_flow(InsightAnalyzer(store).flow("demo"))
    assert "   |\n   v" in text
    # An honest limit: static analysis cannot see runtime registration.
    assert "not an observed runtime trace" in text


def test_rendered_output_stays_ascii(store: GraphStore):
    # This text can reach a Windows console, which is cp1252 here; a box
    # character in it raises UnicodeEncodeError and kills the request.
    analyzer = InsightAnalyzer(store)
    for text in (render_flow(analyzer.flow("demo")), render_hotspots(analyzer.hotspots("demo"))):
        text.encode("cp1252")


# ======================================================== 6. agent behaviour


def test_every_turn_runs_all_three_sources(agent: RepositoryAgent):
    result = ask(agent, "How does login work?")
    assert result["sources"]                      # dense + BM25, via the tool
    assert result["graph_context"].hits           # graph neighbourhood
    assert "REPOSITORY GRAPH" in result["context"]


def test_a_plain_explanation_now_carries_graph_structure(agent: RepositoryAgent):
    # The Phase 3 regression this fixes: explanation used to be retrieval-only,
    # so "how does X connect to Y" never saw the import edge that answers it.
    context = ask(agent, "How does service.py connect to the database?")["context"]
    assert "depends on ->" in context


def test_explorer_selection_answers_a_question_that_names_nothing(agent: RepositoryAgent):
    result = ask(
        agent, "What does this do?",
        explorer={"node_id": NODE("src/auth/service.py"), "file": "src/auth/service.py"},
    )
    assert result["focus_label"] == "service.py"
    assert "service.py" in result["context"]


def test_explorer_context_reaches_the_prompt(agent: RepositoryAgent):
    result = ask(
        agent, "Explain this",
        explorer={"file": "src/auth/service.py", "symbol": "AuthService"},
    )
    assert "CURRENT EXPLORER CONTEXT" in result["context"]
    assert "AuthService" in result["context"]


def test_a_named_file_outranks_the_explorer_selection(agent: RepositoryAgent):
    # The user said a name; that is a more specific instruction than a click.
    result = ask(
        agent, "Tell me about routes.py",
        explorer={"node_id": NODE("src/auth/service.py")},
    )
    assert result["focus_label"] == "routes.py"


def test_the_explorer_selection_outranks_a_stale_carried_focus(agent: RepositoryAgent):
    ask(agent, "Tell me about service.py", conversation_id="c-explorer")
    result = ask(
        agent, "What does this do?", conversation_id="c-explorer",
        explorer={"node_id": NODE("src/api/routes.py")},
    )
    # Clicking a different node is a more recent statement than the last question.
    assert result["focus_label"] == "routes.py"


def test_the_conversation_is_replayed_into_context(agent: RepositoryAgent):
    ask(agent, "Tell me about service.py", conversation_id="c-replay")
    result = ask(agent, "And what depends on it?", conversation_id="c-replay")
    assert "CONVERSATION SO FAR" in result["context"]
    assert "Tell me about service.py" in result["context"]


def test_conversation_memory_is_bounded(agent: RepositoryAgent):
    for i in range(12):
        ask(agent, f"Question {i} about service.py", conversation_id="c-bounded")
    assert len(ask(agent, "And now?", conversation_id="c-bounded")["turns"]) <= 8


def test_a_flow_answer_offers_the_flow_as_the_target(agent: RepositoryAgent):
    # The subject of a flow question is a chain of files the question never
    # names, so seed resolution finds nothing. Without an explicit branch the
    # most visual answer the system produces offered no way to see it.
    navigation = ask(agent, "How does a login request flow through this repository?")[
        "navigation"
    ]
    assert navigation["available"] is True
    assert navigation["focus_mode"] == "flow"
    # Ordered: the graph numbers these as steps.
    assert navigation["node_id"] == NODE("src/app/main.py")
    assert navigation["related_node_ids"]


def test_a_hotspot_answer_offers_the_hotspots(agent: RepositoryAgent):
    navigation = ask(agent, "Which files are most depended on?")["navigation"]
    assert navigation["available"] is True
    assert navigation["node_id"] == NODE("src/auth/service.py")


def test_flow_questions_produce_a_derived_path(agent: RepositoryAgent):
    result = ask(agent, "How does a request flow through this repository?")
    assert result["intent"] == INTENT_FLOW
    assert "   |\n   v" in result["context"]
    assert result["ui_action"]["type"] == "show_flow"


def test_concept_questions_gather_by_theme(agent: RepositoryAgent):
    result = ask(agent, "Show me the database layer")
    assert result["intent"] == INTENT_CONCEPT
    assert "database.py" in result["context"]


def test_hotspot_questions_use_computed_metrics(agent: RepositoryAgent):
    result = ask(agent, "Which files are most depended on?")
    assert result["intent"] == INTENT_HOTSPOT
    assert "import edges" in result["context"]


def test_retrieval_failure_degrades_to_a_graph_answer(service: GraphService):
    class BrokenSearch:
        def run(self, **kwargs):
            raise RuntimeError("chroma is down")

        def invalidate(self, repository=None):
            pass

    broken = RepositoryAgent(
        graph_service=service,
        llm_service=LLMService(llm=FakeListChatModel(responses=["ok"] * 5)),
        search_tool=BrokenSearch(),
        file_tool=FileTool(service),
        architecture_tool=ArchitectureTool(service),
    )
    result = broken.ask("What depends on service.py?", repository="demo")
    # A dead vector store should cost the excerpts, not the whole turn.
    assert result["answer"] == "ok"
    assert "REPOSITORY GRAPH" in result["context"]


class CountingChain:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        return self.inner.invoke(payload)


@pytest.mark.parametrize(
    "question",
    [
        "How does a request flow through this repository?",
        "Show me the database layer",
        "Which files are most depended on?",
        "Rewrite service.py",
        "What does this do?",
    ],
)
def test_the_new_intents_keep_the_one_call_budget(agent: RepositoryAgent, question: str):
    # Generation costs ~150s on the target hardware. Every Phase 4 addition is
    # deterministic code precisely so this stays at one.
    counting = CountingChain(agent.llm.chain)
    agent.llm.chain = counting
    ask(agent, question)
    assert counting.calls == 1


# ======================================================== 7. HTTP endpoints


def test_commits_are_returned_as_timeline_dots(client: TestClient):
    body = client.get("/graph/demo/commits").json()
    assert len(body["commits"]) == len(COMMITS)
    first = body["commits"][0]
    assert {"sha", "author", "authored_at", "summary", "files_changed"} <= set(first)


def test_commits_are_ordered_oldest_first_for_drawing(client: TestClient):
    stamps = [c["authored_at"] for c in client.get("/graph/demo/commits").json()["commits"]]
    assert stamps == sorted(stamps)


def test_a_tagged_commit_carries_its_release(client: TestClient):
    commits = client.get("/graph/demo/commits").json()["commits"]
    assert next(c for c in commits if c["sha"] == "cccc3333")["release"] == "v1.0"


def test_commit_detail_lists_the_files_changed(client: TestClient):
    body = client.get("/graph/demo/commit", params={"sha": "dddd4444"}).json()
    assert body["files_changed"] == 2
    assert {f["relative_path"] for f in body["files"]} == {
        "src/api/routes.py", "src/auth/service.py",
    }


def test_commit_files_resolve_to_graph_nodes(client: TestClient):
    body = client.get("/graph/demo/commit", params={"sha": "cccc3333"}).json()
    assert body["files"][0]["node_id"] == NODE("src/auth/service.py")


def test_an_abbreviated_sha_resolves(client: TestClient):
    assert client.get("/graph/demo/commit", params={"sha": "cccc"}).status_code == 200


def test_an_unknown_commit_is_a_404(client: TestClient):
    assert client.get("/graph/demo/commit", params={"sha": "ffff9999"}).status_code == 404


def test_selecting_a_commit_returns_repository_state(client: TestClient):
    body = client.get("/graph/demo/commit/snapshot", params={"sha": "bbbb2222"}).json()
    assert body["commit"]["short_sha"] == "bbbb2222"
    # Modules first committed after this point are not in the snapshot.
    assert "src/api" not in {n["name"] for n in body["nodes"]}


def test_changes_between_two_commits_are_reported(client: TestClient):
    body = client.get(
        "/graph/demo/changes", params={"from_sha": "bbbb2222", "to_sha": "eeee5555"}
    ).json()
    assert body["files_touched"] > 0
    assert body["modules"]


def test_the_commit_order_of_a_range_does_not_matter(client: TestClient):
    forward = client.get(
        "/graph/demo/changes", params={"from_sha": "bbbb2222", "to_sha": "eeee5555"}
    ).json()
    backward = client.get(
        "/graph/demo/changes", params={"from_sha": "eeee5555", "to_sha": "bbbb2222"}
    ).json()
    assert forward["files_touched"] == backward["files_touched"]


def test_hotspots_are_served_with_their_reasons(client: TestClient):
    hotspots = client.get("/graph/demo/hotspots").json()["hotspots"]
    assert hotspots
    assert all(h["reason"] for h in hotspots)


def test_flow_is_served_as_ordered_steps(client: TestClient):
    body = client.get("/graph/demo/flow").json()
    assert [s["depth"] for s in body["steps"]] == list(range(len(body["steps"])))


def test_a_node_set_spanning_levels_is_served_whole(client: TestClient):
    # A flow crosses directories: src/app/main.py and src/api/routes.py have
    # different parents, so expanding either one's parent shows part of the
    # chain and hides the rest. The set has to be fetchable by name.
    ids = [NODE("src/app/main.py"), NODE("src/api/routes.py"), NODE("src/auth/service.py")]
    body = client.get("/graph/nodes", params=[("id", i) for i in ids]).json()
    assert [n["id"] for n in body["nodes"]] == ids


def test_a_node_set_carries_the_edges_among_it(client: TestClient):
    ids = [NODE("src/api/routes.py"), NODE("src/auth/service.py")]
    body = client.get("/graph/nodes", params=[("id", i) for i in ids]).json()
    assert body["edges"]
    present = {n["id"] for n in body["nodes"]}
    for edge in body["edges"]:
        assert edge["source"] in present and edge["target"] in present


def test_a_node_set_ignores_ids_that_do_not_exist(client: TestClient):
    body = client.get(
        "/graph/nodes", params=[("id", NODE("src/app/main.py")), ("id", "demo::file::gone.py")]
    ).json()
    assert len(body["nodes"]) == 1
    assert body["requested"] == 2


def test_flow_steps_are_layers_not_a_sequence(store: GraphStore):
    # main.py imports routes.py; routes.py imports service.py. Anything at the
    # same depth is concurrent, and the UI numbers by depth so it cannot imply
    # an order the import graph does not contain.
    flow = InsightAnalyzer(store).flow("demo")
    depths = {n["name"]: s.depth for s in flow.steps for n in s.nodes}
    assert depths["main.py"] == 0
    assert depths["routes.py"] == 1
    assert depths["service.py"] == 2


def test_the_neighbourhood_endpoint_is_bounded(client: TestClient):
    body = client.get(
        "/graph/node/neighborhood",
        params={"node_id": NODE("src/auth/service.py"), "limit": 4},
    ).json()
    assert len(body["nodes"]) <= 4
    assert body["seeds"] == [NODE("src/auth/service.py")]


def test_the_neighbourhood_of_an_unknown_node_is_a_404(client: TestClient):
    assert client.get(
        "/graph/node/neighborhood", params={"node_id": "demo::file::nope.py"}
    ).status_code == 404


def test_phase3_graph_endpoints_still_work(client: TestClient):
    # Part 20.17: nothing from Phase 3 may break.
    assert client.get("/graph/demo/overview").status_code == 200
    assert client.get("/graph/demo/timeline").status_code == 200
    assert client.get(
        "/graph/node", params={"node_id": NODE("src/auth/service.py")}
    ).status_code == 200
