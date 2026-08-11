"""LCEL pipeline, prompt assembly and answer cleanup (items 10-11)."""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.prompts.system_prompt import SYSTEM_PROMPT, USER_PROMPT
from app.services.llm_service import LLMService, describe_role, strip_reasoning


def chunk(path, symbol="", content="body", chunk_type="function", start=1, end=10):
    return {
        "relative_path": path,
        "symbol": symbol,
        "chunk_type": chunk_type,
        "start_line": start,
        "end_line": end,
        "content": content,
    }


def service_with(responses: list[str]) -> LLMService:
    return LLMService(llm=FakeListChatModel(responses=responses))


# ------------------------------------------------------------------- roles


@pytest.mark.parametrize(
    "path,role",
    [
        ("fastapi/routing.py", "implementation"),
        ("app/services/chat_service.py", "implementation"),
        ("tests/test_routing.py", "test"),
        ("test_main.py", "test"),
        ("docs_src/bigger_applications/tutorial001.py", "example"),
        ("examples/basic.py", "example"),
        ("docs/en/index.md", "documentation"),
    ],
)
def test_role_is_derived_from_path(path: str, role: str):
    assert describe_role(path) == role


# ----------------------------------------------------------------- context


def test_context_labels_provenance():
    context = LLMService.build_context(
        [chunk("fastapi/routing.py", "APIRouter.include_router", "def include_router(): ...")]
    )
    assert "fastapi/routing.py:1-10" in context
    assert "symbol: APIRouter.include_router" in context
    assert "kind: function" in context
    assert "role: implementation" in context


def test_context_numbers_excerpts():
    context = LLMService.build_context([chunk("a.py"), chunk("b.py"), chunk("c.py")])
    assert "[1]" in context and "[2]" in context and "[3]" in context


def test_context_marks_tests_and_examples():
    context = LLMService.build_context(
        [chunk("tests/test_x.py"), chunk("docs_src/tutorial001.py")]
    )
    assert "role: test" in context
    assert "role: example" in context


def test_context_handles_no_excerpts():
    assert "no excerpts" in LLMService.build_context([])


def test_context_survives_missing_metadata():
    context = LLMService.build_context([{"content": "x"}])
    assert "unknown" in context


# ------------------------------------------------------------- reasoning


def test_think_block_is_stripped():
    assert strip_reasoning("<think>weighing options</think>The answer.") == "The answer."


def test_multiline_think_block_is_stripped():
    raw = "<think>\nline one\nline two\n</think>\n\nFinal answer here."
    assert strip_reasoning(raw) == "Final answer here."


def test_unterminated_think_block_is_dropped():
    # A truncated completion can open the block without closing it.
    assert strip_reasoning("Partial answer.<think>cut off mid-thought") == "Partial answer."


def test_text_without_reasoning_is_untouched():
    assert strip_reasoning("Just an answer.") == "Just an answer."


def test_strip_reasoning_handles_empty():
    assert strip_reasoning("") == ""


# ---------------------------------------------------------------- pipeline


def test_chain_is_lcel_composed():
    service = service_with(["ok"])
    # prompt | llm | parser — each stage must be present and composable.
    assert hasattr(service.chain, "invoke")
    assert hasattr(service.chain, "stream")


def test_generate_response_returns_clean_text():
    service = service_with(["<think>hmm</think>Routing is handled by APIRouter."])
    answer = service.generate_response("How does routing work?", [chunk("fastapi/routing.py")])
    assert answer == "Routing is handled by APIRouter."


def test_generate_response_short_circuits_without_context():
    service = service_with(["should not be used"])
    answer = service.generate_response("anything", [])
    assert "couldn't find" in answer


def test_streaming_yields_chunks():
    service = service_with(["streamed answer"])
    tokens = list(service.stream_response("q", [chunk("a.py")]))
    assert "".join(tokens)


def test_prompt_carries_repository_and_question():
    messages = LLMService(llm=FakeListChatModel(responses=["x"])).prompt.format_messages(
        question="How does routing work?", repository="fastapi", context="ctx"
    )
    rendered = "\n".join(m.content for m in messages)
    assert "fastapi" in rendered
    assert "How does routing work?" in rendered
    assert "ctx" in rendered


# ------------------------------------------------------------------ prompt


def test_system_prompt_states_the_precedence_order():
    lowered = SYSTEM_PROMPT.lower()
    for term in ("implementation", "tests", "example", "documentation"):
        assert term in lowered


def test_system_prompt_forbids_invention():
    lowered = SYSTEM_PROMPT.lower()
    assert "never invent" in lowered


def test_system_prompt_defines_the_no_answer_response():
    assert "couldn't find that in the indexed repository" in SYSTEM_PROMPT


def test_user_prompt_has_the_expected_slots():
    for slot in ("{repository}", "{context}", "{question}"):
        assert slot in USER_PROMPT
