"""Intent routing, tool dispatch and conversational state (Phase 3, stage 3.5).

The LLM is faked throughout: these tests verify routing and context handling,
not generation quality.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.agent.graph import RepositoryAgent
from app.agent.intents import (
    INTENT_ARCHITECTURE,
    INTENT_CODE_EXPLANATION,
    INTENT_CODE_GENERATION,
    INTENT_DEPENDENCY,
    INTENT_FILE,
    INTENT_HISTORY,
    INTENT_READONLY,
    INTENT_SEARCH,
    INTENT_STRUCTURE,
    classify,
    extract_mentions,
)
from app.graph.builder import GraphBuilder
from app.graph.store import GraphStore
from app.services.graph_service import GraphService
from app.services.llm_service import LLMService
from app.tools.architecture import ArchitectureTool
from app.tools.file_tool import FileTool

from .conftest import write

SERVICE_PY = '''\
from .helpers import assist


class AuthService:
    """Handles authentication."""

    def login(self, user):
        return assist(user)
'''

FILES = [
    {"repository": "demo", "relative_path": "src/auth/service.py", "file_name": "service.py",
     "language": "python", "extension": ".py", "file_hash": "h", "content": SERVICE_PY},
    {"repository": "demo", "relative_path": "src/auth/helpers.py", "file_name": "helpers.py",
     "language": "python", "extension": ".py", "file_hash": "h",
     "content": "def assist(user):\n    return user\n"},
    {"repository": "demo", "relative_path": "src/api/routes.py", "file_name": "routes.py",
     "language": "python", "extension": ".py", "file_hash": "h",
     "content": "from src.auth.service import AuthService\n"},
]


class FakeSearchTool:
    """Records calls and returns one deterministic chunk."""

    def __init__(self):
        self.calls: list[dict] = []

    def run(self, query, repository, top_k=5, language=None, chunk_type=None):
        self.calls.append({"query": query, "repository": repository, "top_k": top_k})
        return [
            {
                "relative_path": "src/auth/service.py",
                "file_name": "service.py",
                "repository": repository,
                "symbol": "AuthService.login",
                "chunk_type": "method",
                "chunk_index": 0,
                "start_line": 1,
                "end_line": 8,
                "language": "python",
                "distance": 0.2,
                "score": 0.8,
                "vector_score": 0.8,
                "keyword_score": 0.5,
                "content": SERVICE_PY,
            }
        ]

    def invalidate(self, repository=None):
        pass


@pytest.fixture
def agent(tmp_path: Path) -> RepositoryAgent:
    checkout = tmp_path / "demo"
    for record in FILES:
        write(checkout, record["relative_path"], record["content"])

    store = GraphStore(path=str(tmp_path / "graph.sqlite3"))
    builder = GraphBuilder("demo")
    builder.build(FILES)
    store.replace_repository("demo", builder.graph.nodes, builder.graph.edges)
    store.set_meta("demo", "path", {"value": str(checkout)})
    store.replace_history(
        "demo",
        [{"sha": "a1", "author": "Ada", "authored_at": 1_000, "summary": "add auth", "is_merge": False}],
        [{"sha": "a1", "relative_path": "src/auth/service.py", "module": "src/auth",
          "change_type": "A", "authored_at": 1_000}],
        [{"name": "v1.0", "kind": "tag", "sha": "a1", "created_at": 1_000}],
    )

    graph_service = GraphService(graph_store=store, repositories_root=str(tmp_path))
    llm = LLMService(llm=FakeListChatModel(responses=["A grounded answer."] * 40))

    return RepositoryAgent(
        graph_service=graph_service,
        llm_service=llm,
        search_tool=FakeSearchTool(),
        file_tool=FileTool(graph_service),
        architecture_tool=ArchitectureTool(graph_service),
    )


def ask(agent: RepositoryAgent, question: str, conversation_id: str = "t") -> dict:
    return agent.ask(question, repository="demo", conversation_id=conversation_id)


# ------------------------------------------------------------ classification


@pytest.mark.parametrize(
    "question,intent",
    [
        ("How does routing work?", INTENT_CODE_EXPLANATION),
        ("Where is authentication implemented?", INTENT_SEARCH),
        ("Explain the architecture.", INTENT_ARCHITECTURE),
        ("What depends on AuthService?", INTENT_DEPENDENCY),
        ("Show me the repository structure.", INTENT_STRUCTURE),
        ("What changed in authentication recently?", INTENT_HISTORY),
        ("Create tests for this function.", INTENT_CODE_GENERATION),
        ("Tell me about service.py", INTENT_FILE),
    ],
)
def test_spec_examples_route_correctly(question: str, intent: str):
    assert classify(question).intent == intent


def test_confidence_is_reported():
    assert 0.0 <= classify("What depends on AuthService?").confidence <= 1.0


def test_unmatched_question_defaults_to_explanation_with_low_confidence():
    result = classify("bananas")
    assert result.intent == INTENT_CODE_EXPLANATION
    assert result.confidence < 0.5


def test_imperative_verbs_are_not_treated_as_symbols():
    # "Create tests..." must not resolve focus onto a symbol named `Create`.
    assert "Create" not in extract_mentions("Create tests for this function.")


def test_file_and_symbol_mentions_are_extracted():
    assert "routing.py" in extract_mentions("Tell me about routing.py")
    assert "AuthService" in extract_mentions("What depends on AuthService?")


def test_pronoun_marks_need_for_context():
    assert classify("What does it depend on?").needs_context


# ------------------------------------------------------------------ routing


def test_explanation_uses_retrieval(agent: RepositoryAgent):
    result = ask(agent, "How does login work?")
    assert result["intent"] == INTENT_CODE_EXPLANATION
    assert result["sources"]
    assert agent.search.calls


def test_architecture_uses_the_graph_not_only_retrieval(agent: RepositoryAgent):
    result = ask(agent, "Explain the architecture.")
    assert result["intent"] == INTENT_ARCHITECTURE
    assert "Modules" in result["context"]
    assert "src/auth" in result["context"]


def test_structure_returns_a_tree_and_a_ui_action(agent: RepositoryAgent):
    result = ask(agent, "Show me the repository structure.")
    assert "src/auth/" in result["context"]
    assert result["ui_action"]["type"] == "show_structure"


def test_dependency_analysis_returns_both_directions(agent: RepositoryAgent):
    result = ask(agent, "What depends on service.py?")
    assert result["intent"] == INTENT_DEPENDENCY
    assert "Used by" in result["context"]
    assert result["ui_action"]["type"] == "highlight_dependencies"
    assert result["ui_action"]["dependents"]


def test_dependency_without_a_target_falls_back_to_search(agent: RepositoryAgent):
    result = ask(agent, "what depends on things")
    assert result["sources"]


def test_file_analysis_opens_the_file(agent: RepositoryAgent):
    result = ask(agent, "Tell me about service.py")
    assert result["intent"] == INTENT_FILE
    assert result["ui_action"]["type"] == "open_file"
    assert "symbols:" in result["context"]


def test_history_uses_git_data(agent: RepositoryAgent):
    result = ask(agent, "What changed recently?")
    assert result["intent"] == INTENT_HISTORY
    assert "commits" in result["context"].lower()


def test_change_requests_are_declined_not_fulfilled(agent: RepositoryAgent):
    # Phase 4, Part 16: Repoint is read-only. Phase 3 answered this intent by
    # gathering the target's source and asking the model for new code; that is
    # deliberately gone. The turn still runs — it gathers the same evidence —
    # but the instruction is to explain the change, never to write it.
    result = ask(agent, "Create tests for service.py")
    assert result["intent"] == INTENT_READONLY

    facts = "\n".join(result["facts"])
    assert "read-only" in facts
    assert "Do not produce a patch" in facts


def test_a_declined_request_still_gets_real_evidence(agent: RepositoryAgent):
    # Declining is not the same as being unhelpful: the answer should be able
    # to name what the change would touch, so retrieval and the graph still run.
    result = ask(agent, "Refactor service.py to use a repository pattern")
    assert result["sources"]
    assert "class AuthService" in result["context"]


def test_a_declined_request_offers_no_exploration_target(agent: RepositoryAgent):
    # Part 5: an [Explore this] button under a refusal points nowhere useful.
    result = ask(agent, "Rewrite service.py")
    assert result["navigation"]["available"] is False


def test_search_intent_retrieves(agent: RepositoryAgent):
    result = ask(agent, "Where is authentication implemented?")
    assert result["intent"] == INTENT_SEARCH
    assert result["sources"]


# ---------------------------------------------------- conversational context


def test_follow_up_resolves_it_to_the_previous_focus(agent: RepositoryAgent):
    first = ask(agent, "Tell me about service.py", conversation_id="c1")
    assert first["focus_label"] == "service.py"

    second = ask(agent, "What does it depend on?", conversation_id="c1")
    assert second["intent"] == INTENT_DEPENDENCY
    # "it" carried the focus forward rather than falling back to search.
    assert second["focus_label"] == "service.py"
    assert "Depends on" in second["context"]


def test_new_mention_replaces_the_focus(agent: RepositoryAgent):
    ask(agent, "Tell me about service.py", conversation_id="c2")
    second = ask(agent, "Tell me about helpers.py", conversation_id="c2")
    assert second["focus_label"] == "helpers.py"


def test_conversations_are_isolated(agent: RepositoryAgent):
    ask(agent, "Tell me about service.py", conversation_id="a")
    other = ask(agent, "What does it depend on?", conversation_id="b")
    # A different conversation has no prior focus to inherit.
    assert other["focus_label"] is None


def test_trace_records_the_path_taken(agent: RepositoryAgent):
    result = ask(agent, "What depends on service.py?")
    nodes = [step["node"] for step in result["trace"]]
    assert nodes[:2] == ["resolve_context", "classify"]
    assert "dependencies" in nodes
    assert nodes[-1] == "answer"


def test_answer_is_generated(agent: RepositoryAgent):
    assert ask(agent, "How does login work?")["answer"] == "A grounded answer."


class CountingChain:
    """Wraps the LCEL chain to count invocations.

    The chain is a Pydantic ``RunnableSequence`` and rejects attribute
    assignment, so the whole object is swapped rather than its method patched.
    """

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        return self.inner.invoke(payload)


@pytest.mark.parametrize(
    "question",
    ["Explain the architecture.", "What depends on service.py?", "How does login work?"],
)
def test_only_one_llm_call_per_turn(agent: RepositoryAgent, question: str):
    # Latency budget: generation costs ~150s on the target hardware, so routing
    # and gathering stay deterministic — exactly one LLM call per turn,
    # whichever intent is taken.
    counting = CountingChain(agent.llm.chain)
    agent.llm.chain = counting

    ask(agent, question)
    assert counting.calls == 1
