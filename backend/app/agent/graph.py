"""LangGraph routing for repository questions.

Shape::

    resolve_context -> classify -> [one analysis node] -> generate

One analysis node runs per turn, chosen by the router. There is no agent loop
and no tool-calling LLM: on this hardware every LLM call costs ~150 seconds, so
the graph spends exactly one on the answer and does all its routing and
gathering with deterministic code.

State is checkpointed per conversation, which is what makes "what does it
depend on?" resolvable — the previous turn's focus is still in state.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agent.intents import (
    INTENT_ARCHITECTURE,
    INTENT_CODE_EXPLANATION,
    INTENT_CODE_GENERATION,
    INTENT_DEPENDENCY,
    INTENT_FILE,
    INTENT_HISTORY,
    INTENT_SEARCH,
    INTENT_STRUCTURE,
    classify,
)
from app.graph.model import NODE_FILE
from app.services.graph_service import GraphNotFoundError, GraphService
from app.services.llm_service import LLMService
from app.tools.architecture import ArchitectureTool
from app.tools.file_tool import FileTool
from app.tools.search_tool import SearchTool

logger = logging.getLogger(__name__)


def _last(current: Any, incoming: Any) -> Any:
    """Reducer: the newest non-None value wins."""
    return incoming if incoming is not None else current


class AgentState(TypedDict, total=False):
    """Everything a turn reads or writes.

    ``focus_node_id`` and ``focus_label`` survive between turns — they are what
    "it" and "this" resolve to.
    """

    question: Annotated[str, _last]
    repository: Annotated[str, _last]
    top_k: Annotated[int, _last]

    intent: Annotated[str, _last]
    confidence: Annotated[float, _last]
    mentions: Annotated[list, _last]

    focus_node_id: Annotated[str | None, _last]
    focus_label: Annotated[str | None, _last]

    context: Annotated[str, _last]
    sources: Annotated[list, _last]
    graph_payload: Annotated[dict | None, _last]
    ui_action: Annotated[dict | None, _last]

    answer: Annotated[str, _last]
    trace: Annotated[list, _last]


class RepositoryAgent:
    """Compiled LangGraph app plus the tools its nodes call."""

    def __init__(
        self,
        graph_service: GraphService | None = None,
        llm_service: LLMService | None = None,
        search_tool: SearchTool | None = None,
        file_tool: FileTool | None = None,
        architecture_tool: ArchitectureTool | None = None,
        checkpointer=None,
    ):
        self.graph_service = graph_service or GraphService()
        self.llm = llm_service or LLMService()
        self.search = search_tool or SearchTool()
        self.files = file_tool or FileTool(self.graph_service)
        self.architecture = architecture_tool or ArchitectureTool(self.graph_service)

        # In-process memory. A durable checkpointer (SQLite/Postgres) drops in
        # here unchanged when conversations need to outlive the process.
        self.checkpointer = checkpointer or MemorySaver()
        self.app = self._build()

    # -- graph -------------------------------------------------------------

    def _build(self):
        builder = StateGraph(AgentState)

        builder.add_node("resolve_context", self.resolve_context)
        builder.add_node("classify", self.classify_intent)
        builder.add_node("search", self.node_search)
        builder.add_node("explain", self.node_explain)
        builder.add_node("architecture", self.node_architecture)
        builder.add_node("structure", self.node_structure)
        builder.add_node("dependencies", self.node_dependencies)
        builder.add_node("file", self.node_file)
        builder.add_node("history", self.node_history)
        builder.add_node("generate_code", self.node_generate_code)
        builder.add_node("answer", self.node_answer)

        builder.set_entry_point("resolve_context")
        builder.add_edge("resolve_context", "classify")

        builder.add_conditional_edges(
            "classify",
            self.route,
            {
                INTENT_SEARCH: "search",
                INTENT_CODE_EXPLANATION: "explain",
                INTENT_ARCHITECTURE: "architecture",
                INTENT_STRUCTURE: "structure",
                INTENT_DEPENDENCY: "dependencies",
                INTENT_FILE: "file",
                INTENT_HISTORY: "history",
                INTENT_CODE_GENERATION: "generate_code",
            },
        )

        for node in (
            "search", "explain", "architecture", "structure",
            "dependencies", "file", "history", "generate_code",
        ):
            builder.add_edge(node, "answer")
        builder.add_edge("answer", END)

        return builder.compile(checkpointer=self.checkpointer)

    @staticmethod
    def route(state: AgentState) -> str:
        return state.get("intent") or INTENT_CODE_EXPLANATION

    # -- nodes -------------------------------------------------------------

    def resolve_context(self, state: AgentState) -> dict:
        """Carry the previous turn's focus forward.

        Checkpointed state already contains the old focus; this node only has
        to decide whether the new question replaces it. A question naming a
        concrete file or symbol re-focuses; a pronoun keeps the old one.
        """
        question = state.get("question", "")
        classification = classify(question)
        repository = state.get("repository", "")

        focus_id = state.get("focus_node_id")
        focus_label = state.get("focus_label")

        if classification.mentions and repository:
            for mention in classification.mentions:
                matches = self.files.find(repository, mention, limit=1)
                if matches:
                    focus_id = matches[0]["id"]
                    focus_label = matches[0]["name"]
                    break

        trace = list(state.get("trace") or [])
        trace.append(
            {
                "node": "resolve_context",
                "mentions": classification.mentions,
                "needs_context": classification.needs_context,
                "focus": focus_label,
                "carried_over": bool(
                    classification.needs_context and not classification.mentions
                ),
            }
        )
        return {"focus_node_id": focus_id, "focus_label": focus_label, "trace": trace}

    def classify_intent(self, state: AgentState) -> dict:
        classification = classify(state.get("question", ""))
        trace = list(state.get("trace") or [])
        trace.append(
            {
                "node": "classify",
                "intent": classification.intent,
                "confidence": classification.confidence,
                "scores": classification.scores,
            }
        )
        logger.info(
            "intent=%s confidence=%.2f question=%r",
            classification.intent, classification.confidence, state.get("question", "")[:70],
        )
        return {
            "intent": classification.intent,
            "confidence": classification.confidence,
            "mentions": classification.mentions,
            "trace": trace,
        }

    # -- retrieval-backed nodes --

    def _retrieve(self, state: AgentState, top_k: int | None = None) -> list[dict]:
        return self.search.run(
            query=state.get("question", ""),
            repository=state.get("repository", ""),
            top_k=top_k or state.get("top_k") or 5,
        )

    def node_search(self, state: AgentState) -> dict:
        sources = self._retrieve(state)
        return self._with_trace(state, "search", {"sources": sources, "context": ""}, len(sources))

    def node_explain(self, state: AgentState) -> dict:
        sources = self._retrieve(state)
        return self._with_trace(state, "explain", {"sources": sources, "context": ""}, len(sources))

    # -- graph-backed nodes --

    def node_architecture(self, state: AgentState) -> dict:
        repository = state.get("repository", "")
        try:
            context = self.architecture.describe_architecture(repository)
        except GraphNotFoundError as exc:
            context = f"No repository graph available: {exc}"
        # Retrieval still runs: the skeleton says what exists, the chunks say
        # how it works, and the answer needs both.
        sources = self._retrieve(state, top_k=max(8, state.get("top_k") or 5))
        return self._with_trace(
            state, "architecture", {"context": context, "sources": sources}, len(sources)
        )

    def node_structure(self, state: AgentState) -> dict:
        repository = state.get("repository", "")
        try:
            context = self.architecture.describe_structure(repository)
            payload = self.graph_service.overview(repository)
            action = {"type": "show_structure", "repository": repository}
        except GraphNotFoundError as exc:
            context, payload, action = f"No repository graph available: {exc}", None, None
        return self._with_trace(
            state,
            "structure",
            {"context": context, "sources": [], "graph_payload": payload, "ui_action": action},
            0,
        )

    def node_dependencies(self, state: AgentState) -> dict:
        focus = state.get("focus_node_id")
        if not focus:
            # Nothing to analyse: fall back to retrieval rather than guessing.
            sources = self._retrieve(state)
            return self._with_trace(
                state,
                "dependencies",
                {
                    "context": "No specific file or symbol was identified in the question.",
                    "sources": sources,
                },
                len(sources),
            )

        try:
            context, payload = self.architecture.describe_dependencies(focus)
            action = {
                "type": "highlight_dependencies",
                "node_id": focus,
                "dependencies": [n["id"] for n in payload["dependencies"]],
                "dependents": [n["id"] for n in payload["dependents"]],
            }
        except GraphNotFoundError as exc:
            context, payload, action = f"Could not resolve {focus}: {exc}", None, None

        return self._with_trace(
            state,
            "dependencies",
            {"context": context, "sources": [], "graph_payload": payload, "ui_action": action},
            0,
        )

    def node_file(self, state: AgentState) -> dict:
        focus = state.get("focus_node_id")
        repository = state.get("repository", "")

        if not focus:
            sources = self._retrieve(state)
            return self._with_trace(
                state, "file", {"context": "", "sources": sources}, len(sources)
            )

        try:
            detail = self.files.detail(focus)
            node = detail["node"]
            lines = [
                f"{node['kind']} {node['name']} ({node['key']})",
                f"language: {node['data'].get('language', 'unknown')}, "
                f"lines: {node['data'].get('line_count', '?')}, "
                f"complexity: {node['data'].get('complexity', 'unknown')}",
            ]
            if detail.get("symbols"):
                lines.append(
                    "symbols: " + ", ".join(s["name"] for s in detail["symbols"][:25])
                )
            if detail.get("dependencies"):
                lines.append(
                    "imports: " + ", ".join(d["name"] for d in detail["dependencies"][:15])
                )
            if detail.get("dependents"):
                lines.append(
                    "used by: " + ", ".join(d["name"] for d in detail["dependents"][:15])
                )
            context = "\n".join(lines)
            action = {"type": "open_file", "node_id": focus} if node["kind"] == NODE_FILE else None
        except GraphNotFoundError as exc:
            context, action = f"Could not inspect {focus}: {exc}", None

        sources = self.search.run(
            query=state.get("question", ""), repository=repository,
            top_k=state.get("top_k") or 5,
        )
        return self._with_trace(
            state, "file", {"context": context, "sources": sources, "ui_action": action},
            len(sources),
        )

    def node_history(self, state: AgentState) -> dict:
        repository = state.get("repository", "")
        focus = state.get("focus_node_id")
        try:
            context = self.architecture.describe_history(repository, node_id=focus)
        except GraphNotFoundError as exc:
            context = f"No history available: {exc}"
        return self._with_trace(state, "history", {"context": context, "sources": []}, 0)

    def node_generate_code(self, state: AgentState) -> dict:
        """Gather the target's real code before asking for new code.

        Generation without the actual implementation in context produces
        plausible-looking tests for functions that do not exist.
        """
        focus = state.get("focus_node_id")
        context_parts: list[str] = []

        if focus:
            try:
                node = self.graph_service.store.get_node(focus)
                target = focus
                if node and node["kind"] != NODE_FILE:
                    relative = node["data"].get("relative_path")
                    if relative:
                        file_node = self.graph_service.store.find_node(
                            state.get("repository", ""), relative, NODE_FILE
                        )
                        target = file_node["id"] if file_node else focus
                content = self.files.content(target)
                excerpt = "\n".join(content["content"].splitlines()[:200])
                context_parts.append(
                    f"Current implementation ({content['relative_path']}):\n{excerpt}"
                )
            except GraphNotFoundError:
                pass

        sources = self._retrieve(state, top_k=max(8, state.get("top_k") or 5))
        return self._with_trace(
            state,
            "generate_code",
            {"context": "\n\n".join(context_parts), "sources": sources},
            len(sources),
        )

    # -- answer --

    def node_answer(self, state: AgentState) -> dict:
        """The single LLM call of the turn."""
        question = state.get("question", "")
        repository = state.get("repository", "")
        sources = state.get("sources") or []
        context = state.get("context") or ""

        if not sources and not context.strip():
            return {
                "answer": "I couldn't find that in the indexed repository.",
                "trace": self._trace(state, "answer", {"skipped": True}),
            }

        blocks = []
        if context.strip():
            blocks.append(f"Repository facts (from static analysis and git history):\n{context}")
        if sources:
            blocks.append(self.llm.build_context(sources))

        answer = self.llm.chain.invoke(
            {
                "question": question,
                "repository": repository or "(unspecified)",
                "context": "\n\n---\n\n".join(blocks),
            }
        )
        from app.services.llm_service import strip_reasoning

        return {
            "answer": strip_reasoning(answer),
            "trace": self._trace(state, "answer", {"blocks": len(blocks)}),
        }

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _trace(state: AgentState, node: str, extra: dict) -> list:
        trace = list(state.get("trace") or [])
        trace.append({"node": node, **extra})
        return trace

    def _with_trace(self, state: AgentState, node: str, payload: dict, sources: int) -> dict:
        return {**payload, "trace": self._trace(state, node, {"sources": sources})}

    # -- entry point -------------------------------------------------------

    def ask(
        self,
        question: str,
        repository: str,
        conversation_id: str = "default",
        top_k: int = 5,
    ) -> dict:
        """Run one turn. State persists per ``conversation_id``."""
        config = {"configurable": {"thread_id": f"{repository}:{conversation_id}"}}
        result = self.app.invoke(
            {
                "question": question,
                "repository": repository,
                "top_k": top_k,
                # Reset per-turn outputs; focus deliberately survives.
                "context": "",
                "sources": [],
                "graph_payload": None,
                "ui_action": None,
                "trace": [],
            },
            config=config,
        )
        return result
