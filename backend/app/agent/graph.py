"""LangGraph routing for repository questions.

Shape::

    resolve_context -> classify -> gather -> [one analysis node] -> generate

``gather`` is the Phase 4 addition and the reason the rest of the graph got
simpler. Phase 3 gave each intent its own retrieval call and its own way of
writing a context string, which meant every node re-decided what evidence
looked like and the graph was never consulted unless the router happened to
pick a graph node. Now every turn runs all three sources — dense, BM25 and
graph neighbourhood — and the analysis nodes only add the *extra* structural
facts their intent needs. Fusion, deduplication, cross-ranking and budgeting
happen once, in ``ContextBuilder``.

One LLM call per turn, still. On this hardware generation costs ~150 seconds,
so routing, resolution and gathering are all deterministic code and the single
call is spent on the answer. That is a hard invariant, and a test asserts it.

State is checkpointed per conversation, which is what makes "what does it
depend on?" resolvable — the previous turn's focus is still in state.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agent.context import (
    ContextBuilder,
    ConversationTurn,
    ExplorerContext,
    RepositoryContext,
)
from app.agent.intents import (
    INTENT_ARCHITECTURE,
    INTENT_CODE_EXPLANATION,
    INTENT_CONCEPT,
    INTENT_DEPENDENCY,
    INTENT_FILE,
    INTENT_FLOW,
    INTENT_HISTORY,
    INTENT_HOTSPOT,
    INTENT_READONLY,
    INTENT_SEARCH,
    INTENT_STRUCTURE,
    classify,
)
from app.agent.navigation import NavigationPlanner
from app.graph.model import NODE_FILE
from app.rag.graph_retriever import GraphRetriever
from app.services.graph_service import GraphNotFoundError, GraphService
from app.services.llm_service import LLMService, strip_reasoning
from app.tools.architecture import ArchitectureTool
from app.tools.file_tool import FileTool
from app.tools.search_tool import SearchTool

logger = logging.getLogger(__name__)

#: Retrieval depth per intent. Architecture and concept questions synthesise
#: across many units and need a wider net; a dependency question is answered
#: by the graph and needs only enough code to illustrate it.
_TOP_K_BY_INTENT: dict[str, int] = {
    INTENT_ARCHITECTURE: 10,
    INTENT_CONCEPT: 10,
    INTENT_FLOW: 8,
    INTENT_SEARCH: 8,
    INTENT_DEPENDENCY: 4,
    INTENT_STRUCTURE: 3,
    INTENT_HOTSPOT: 3,
    INTENT_READONLY: 4,
}

#: Intents whose answer is structural. Retrieval still runs for them, but the
#: graph and the facts lead, and the code is illustration.
_STRUCTURAL_INTENTS = frozenset(
    {INTENT_ARCHITECTURE, INTENT_STRUCTURE, INTENT_DEPENDENCY, INTENT_HISTORY,
     INTENT_HOTSPOT, INTENT_FLOW}
)


def _last(current: Any, incoming: Any) -> Any:
    """Reducer: the newest non-None value wins."""
    return incoming if incoming is not None else current


class AgentState(TypedDict, total=False):
    """Everything a turn reads or writes.

    ``focus_node_id``, ``focus_label`` and ``turns`` survive between turns —
    they are what "it" and "this" resolve to.
    """

    question: Annotated[str, _last]
    repository: Annotated[str, _last]
    top_k: Annotated[int, _last]
    #: Serialised ``ExplorerContext``. Arrives per turn from the frontend.
    explorer: Annotated[dict | None, _last]

    intent: Annotated[str, _last]
    confidence: Annotated[float, _last]
    mentions: Annotated[list, _last]
    concepts: Annotated[list, _last]

    focus_node_id: Annotated[str | None, _last]
    focus_label: Annotated[str | None, _last]
    #: Prior exchanges, as ``{question, answer, focus_label}``.
    turns: Annotated[list, _last]

    #: Raw source output, before the context builder sees it.
    chunks: Annotated[list, _last]
    graph_context: Annotated[Any, _last]
    facts: Annotated[list, _last]
    history_text: Annotated[str, _last]
    #: Nodes an analysis node derived as the subject, for navigation.
    navigation_nodes: Annotated[list, _last]

    #: The assembled context, and everything derived from it.
    context: Annotated[str, _last]
    sources: Annotated[list, _last]
    graph_payload: Annotated[dict | None, _last]
    ui_action: Annotated[dict | None, _last]
    navigation: Annotated[dict | None, _last]
    context_stats: Annotated[dict, _last]

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
        graph_retriever: GraphRetriever | None = None,
        context_builder: ContextBuilder | None = None,
        checkpointer=None,
    ):
        self.graph_service = graph_service or GraphService()
        self.llm = llm_service or LLMService()
        self.search = search_tool or SearchTool()
        self.files = file_tool or FileTool(self.graph_service)
        self.architecture = architecture_tool or ArchitectureTool(self.graph_service)
        self.graph_retriever = graph_retriever or GraphRetriever(self.graph_service)
        self.context_builder = context_builder or ContextBuilder()
        self.navigator = NavigationPlanner()

        # In-process memory. A durable checkpointer (SQLite/Postgres) drops in
        # here unchanged when conversations need to outlive the process.
        self.checkpointer = checkpointer or MemorySaver()
        self.app = self._build()

    # -- graph -------------------------------------------------------------

    def _build(self):
        builder = StateGraph(AgentState)

        builder.add_node("resolve_context", self.resolve_context)
        builder.add_node("classify", self.classify_intent)
        builder.add_node("gather", self.gather)
        builder.add_node("search", self.node_search)
        builder.add_node("explain", self.node_explain)
        builder.add_node("architecture", self.node_architecture)
        builder.add_node("structure", self.node_structure)
        builder.add_node("dependencies", self.node_dependencies)
        builder.add_node("file", self.node_file)
        builder.add_node("history", self.node_history)
        builder.add_node("flow", self.node_flow)
        builder.add_node("concept", self.node_concept)
        builder.add_node("hotspots", self.node_hotspots)
        builder.add_node("readonly", self.node_readonly)
        builder.add_node("answer", self.node_answer)

        builder.set_entry_point("resolve_context")
        builder.add_edge("resolve_context", "classify")
        # Every intent gets all three retrieval sources. What differs is the
        # structural analysis layered on top, which is what the router picks.
        builder.add_edge("classify", "gather")

        builder.add_conditional_edges(
            "gather",
            self.route,
            {
                INTENT_SEARCH: "search",
                INTENT_CODE_EXPLANATION: "explain",
                INTENT_ARCHITECTURE: "architecture",
                INTENT_STRUCTURE: "structure",
                INTENT_DEPENDENCY: "dependencies",
                INTENT_FILE: "file",
                INTENT_HISTORY: "history",
                INTENT_FLOW: "flow",
                INTENT_CONCEPT: "concept",
                INTENT_HOTSPOT: "hotspots",
                INTENT_READONLY: "readonly",
            },
        )

        for node in (
            "search", "explain", "architecture", "structure", "dependencies",
            "file", "history", "flow", "concept", "hotspots", "readonly",
        ):
            builder.add_edge(node, "answer")
        builder.add_edge("answer", END)

        return builder.compile(checkpointer=self.checkpointer)

    @staticmethod
    def route(state: AgentState) -> str:
        return state.get("intent") or INTENT_CODE_EXPLANATION

    # -- resolution --------------------------------------------------------

    def resolve_context(self, state: AgentState) -> dict:
        """Decide what this turn is about.

        Three sources of subject, in priority order:

        1. A name in the question — the user said it, so it wins.
        2. The Explorer selection — Part 3's "what does this do?" without
           naming anything.
        3. The previous turn's focus, carried in checkpointed state.

        The Explorer outranks the carried focus deliberately: if the user has
        clicked a different node since the last question, that click is a more
        recent statement of intent than the last thing they asked about.
        """
        question = state.get("question", "")
        classification = classify(question)
        repository = state.get("repository", "")
        explorer = self._explorer(state)

        focus_id = state.get("focus_node_id")
        focus_label = state.get("focus_label")
        source = "carried"

        resolved = None
        for mention in classification.mentions:
            if not repository:
                break
            matches = self.files.find(repository, mention, limit=1)
            if matches:
                resolved, source = matches[0], "mention"
                break

        if resolved is None and explorer.node_id:
            node = self.graph_service.store.get_node(explorer.node_id)
            if node is not None:
                resolved, source = node, "explorer"

        if resolved is not None:
            focus_id, focus_label = resolved["id"], resolved["name"]

        trace = list(state.get("trace") or [])
        trace.append(
            {
                "node": "resolve_context",
                "mentions": classification.mentions,
                "concepts": classification.concepts,
                "needs_context": classification.needs_context,
                "focus": focus_label,
                "focus_source": source,
                "explorer": bool(explorer),
                "carried_over": source == "carried" and bool(focus_id),
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
            "concepts": classification.concepts,
            "trace": trace,
        }

    # -- gathering ---------------------------------------------------------

    def gather(self, state: AgentState) -> dict:
        """Run all three retrieval sources for this turn.

        Text retrieval first, because its results seed the graph walk when the
        question named nothing concrete: a vague question still gets structure
        by asking the graph about whatever the chunks came from.
        """
        question = state.get("question", "")
        repository = state.get("repository", "")
        intent = state.get("intent", "")
        explorer = self._explorer(state)

        top_k = state.get("top_k") or 5
        wanted = max(top_k, _TOP_K_BY_INTENT.get(intent, top_k))

        chunks: list[dict] = []
        if repository:
            try:
                chunks = self.search.run(
                    query=question, repository=repository, top_k=wanted
                )
            except Exception as exc:  # noqa: BLE001 - retrieval must not kill a turn
                # A missing collection or a cold embedding model should degrade
                # to a graph-only answer, not a 500.
                logger.warning("retrieval failed: %s", type(exc).__name__)

        graph_context = self.graph_retriever.retrieve(
            repository=repository,
            mentions=state.get("mentions") or [],
            focus_node_id=state.get("focus_node_id"),
            explorer_node_ids=explorer.seed_ids(),
            chunk_paths=[c.get("relative_path", "") for c in chunks[:5]],
        )

        trace = list(state.get("trace") or [])
        trace.append(
            {
                "node": "gather",
                "chunks": len(chunks),
                "graph_nodes": len(graph_context.hits),
                "graph_edges": len(graph_context.edges),
                "top_k": wanted,
            }
        )
        return {"chunks": chunks, "graph_context": graph_context, "trace": trace}

    # -- analysis nodes ----------------------------------------------------
    #
    # Each adds only the structural facts its intent needs. Retrieval and
    # graph context are already in state; nothing here re-retrieves.

    def node_search(self, state: AgentState) -> dict:
        return self._analysed(state, "search", [])

    def node_explain(self, state: AgentState) -> dict:
        return self._analysed(state, "explain", [])

    def node_architecture(self, state: AgentState) -> dict:
        repository = state.get("repository", "")
        try:
            facts = [self.architecture.describe_architecture(repository)]
        except GraphNotFoundError as exc:
            facts = [f"No repository graph available: {exc}"]
        return self._analysed(state, "architecture", facts)

    def node_structure(self, state: AgentState) -> dict:
        repository = state.get("repository", "")
        payload = None
        action = None
        try:
            facts = [self.architecture.describe_structure(repository)]
            payload = self.graph_service.overview(repository)
            action = {"type": "show_structure", "repository": repository}
        except GraphNotFoundError as exc:
            facts = [f"No repository graph available: {exc}"]
        return self._analysed(
            state, "structure", facts, graph_payload=payload, ui_action=action
        )

    def node_dependencies(self, state: AgentState) -> dict:
        focus = state.get("focus_node_id")
        if not focus:
            # The graph context may still have seeded from retrieval, so this
            # is a missing *subject*, not a missing answer.
            return self._analysed(
                state,
                "dependencies",
                ["No specific file or symbol was identified in the question."],
            )

        payload = None
        action = None
        try:
            text, payload = self.architecture.describe_dependencies(focus)
            facts = [text]
            action = {
                "type": "highlight_dependencies",
                "node_id": focus,
                "dependencies": [n["id"] for n in payload["dependencies"]],
                "dependents": [n["id"] for n in payload["dependents"]],
            }
        except GraphNotFoundError as exc:
            facts = [f"Could not resolve {focus}: {exc}"]
        return self._analysed(
            state, "dependencies", facts, graph_payload=payload, ui_action=action
        )

    def node_file(self, state: AgentState) -> dict:
        focus = state.get("focus_node_id")
        if not focus:
            return self._analysed(state, "file", [])

        action = None
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
            facts = ["\n".join(lines)]
            if node["kind"] == NODE_FILE:
                action = {"type": "open_file", "node_id": focus}
        except GraphNotFoundError as exc:
            facts = [f"Could not inspect {focus}: {exc}"]
        return self._analysed(state, "file", facts, ui_action=action)

    def node_history(self, state: AgentState) -> dict:
        repository = state.get("repository", "")
        focus = state.get("focus_node_id")
        try:
            history = self.architecture.describe_history(repository, node_id=focus)
        except GraphNotFoundError as exc:
            history = f"No history available: {exc}"
        return self._analysed(state, "history", [], history=history)

    def node_flow(self, state: AgentState) -> dict:
        """Part 10: a path through the repository rather than one unit."""
        from app.graph.insights import render_flow

        repository = state.get("repository", "")
        focus = state.get("focus_node_id")
        facts: list[str] = []
        action = None

        node_ids: list[str] = []
        try:
            from app.graph.insights import InsightAnalyzer

            flow = InsightAnalyzer(self.graph_service.store).flow(repository, focus)
            if flow:
                facts.append(render_flow(flow))
                node_ids = flow.node_ids
                action = {"type": "show_flow", "node_ids": node_ids}
        except GraphNotFoundError as exc:
            facts.append(f"No repository graph available: {exc}")

        if not facts:
            facts.append(
                "No entry point could be identified in the import graph, so no "
                "flow was derived. The answer must come from the retrieved code."
            )
        return self._analysed(
            state, "flow", facts, ui_action=action, navigation_nodes=node_ids
        )

    def node_concept(self, state: AgentState) -> dict:
        """Part 8: a theme spread across files rather than one named symbol.

        Concept terms get their own graph search because the question names
        no node — "the database layer" is not a symbol, but the files that
        implement it are findable by name, and their neighbourhoods are what
        the answer needs.
        """
        repository = state.get("repository", "")
        concepts = state.get("concepts") or []
        facts: list[str] = []

        matches: list[dict] = []
        for concept in concepts[:3]:
            try:
                matches.extend(self.graph_service.search(repository, concept, limit=12))
            except GraphNotFoundError:
                break

        if matches:
            seen: set[str] = set()
            lines = [
                f"Nodes whose name or path mentions {' / '.join(concepts)} "
                f"({len(matches)} matches, closest first):"
            ]
            for node in matches[:20]:
                if node["id"] in seen:
                    continue
                seen.add(node["id"])
                lines.append(f"  {node['kind']} {node['name']}  [{node['key']}]")
            facts.append("\n".join(lines))

        return self._analysed(state, "concept", facts)

    def node_hotspots(self, state: AgentState) -> dict:
        """Part 13: computed metrics, each carrying the count behind it."""
        from app.graph.insights import InsightAnalyzer, render_hotspots

        repository = state.get("repository", "")
        node_ids: list[str] = []
        try:
            hotspots = InsightAnalyzer(self.graph_service.store).hotspots(repository)
            facts = [render_hotspots(hotspots)] if hotspots else []
            # Structural hotspots only: churn and complexity rank the same
            # files by other measures, and offering to explore one file three
            # times is not three destinations.
            node_ids = [h.node_id for h in hotspots if h.category == "structural"][:8]
        except GraphNotFoundError as exc:
            facts = [f"No repository graph available: {exc}"]

        if not facts:
            facts = [
                "No node in this repository crosses the thresholds for a hotspot "
                "(at least 3 dependents, or 3 commits in the indexed window)."
            ]
        return self._analysed(state, "hotspots", facts, navigation_nodes=node_ids)

    def node_readonly(self, state: AgentState) -> dict:
        """Part 16: decline the change, then explain what it would involve.

        Phase 3 answered these by generating code. Phase 4 does not: Repoint
        is a read-only intelligence tool, and the honest response is to say so
        and then be useful in the way the product actually is — by explaining
        what the change would touch. The graph context gathered upstream
        already knows that, so the answer can be specific rather than a
        blanket refusal.
        """
        facts = [
            "The user has asked for a change to the repository. Repoint is "
            "read-only: it never writes, edits, commits or refactors code.\n"
            "Answer by (1) saying plainly in one sentence that Repoint does not "
            "modify repositories, then (2) being useful anyway — explain what "
            "the change would involve, which files and symbols it would touch "
            "(name them from the evidence below), and what to watch out for. "
            "Do not produce a patch, a diff, or a full replacement file."
        ]
        return self._analysed(state, "readonly", facts)

    # -- answer ------------------------------------------------------------

    def node_answer(self, state: AgentState) -> dict:
        """Assemble the context, decide navigation, make the one LLM call."""
        context = self._context(state)
        navigation = self.navigator.decide(context)

        if context.is_empty():
            return {
                "answer": "I couldn't find that in the indexed repository.",
                "context": "",
                "sources": [],
                "navigation": navigation.to_dict(),
                "context_stats": context.stats,
                "trace": self._trace(state, "answer", {"skipped": True}),
            }

        answer = self.llm.chain.invoke(
            {
                "question": context.query,
                "repository": context.repository or "(unspecified)",
                "context": context.render(),
            }
        )

        return {
            "answer": strip_reasoning(answer),
            "context": context.render(),
            "sources": context.code,
            "navigation": navigation.to_dict(),
            "context_stats": context.stats,
            "turns": self._record_turn(state, strip_reasoning(answer)),
            "trace": self._trace(
                state,
                "answer",
                {"navigation": navigation.target_type or "none", **context.stats},
            ),
        }

    # -- helpers -----------------------------------------------------------

    def _context(self, state: AgentState) -> RepositoryContext:
        """Build the turn's ``RepositoryContext`` from gathered state."""
        intent = state.get("intent", "")
        return self.context_builder.build(
            query=state.get("question", ""),
            repository=state.get("repository", ""),
            intent=intent,
            chunks=state.get("chunks") or [],
            graph=state.get("graph_context"),
            explorer=self._explorer(state),
            conversation=[
                ConversationTurn(
                    question=turn.get("question", ""),
                    answer=turn.get("answer", ""),
                    focus_label=turn.get("focus_label", ""),
                )
                for turn in (state.get("turns") or [])
            ],
            facts=state.get("facts") or [],
            history=state.get("history_text") or "",
            navigation_nodes=state.get("navigation_nodes") or [],
            # Structural answers lean on facts, so they spend less of the
            # budget on excerpts and leave room for the graph.
            code_budget=4 if intent in _STRUCTURAL_INTENTS else None,
        )

    @staticmethod
    def _explorer(state: AgentState) -> ExplorerContext:
        raw = state.get("explorer") or {}
        if not isinstance(raw, dict):
            return ExplorerContext()
        return ExplorerContext(
            repository=raw.get("repository", "") or "",
            module=raw.get("module", "") or "",
            file=raw.get("file", "") or "",
            symbol=raw.get("symbol", "") or "",
            node_id=raw.get("node_id", "") or "",
            focus_node_ids=list(raw.get("focus_node_ids") or []),
            commit_sha=raw.get("commit_sha", "") or "",
            timestamp=raw.get("timestamp"),
        )

    @staticmethod
    def _record_turn(state: AgentState, answer: str) -> list:
        """Append this exchange to the replayed conversation.

        Trimmed to the last few turns here rather than at render time so the
        checkpoint does not grow without bound over a long conversation.
        """
        turns = list(state.get("turns") or [])
        turns.append(
            {
                "question": state.get("question", ""),
                "answer": answer,
                "focus_label": state.get("focus_label") or "",
            }
        )
        return turns[-8:]

    @staticmethod
    def _trace(state: AgentState, node: str, extra: dict) -> list:
        trace = list(state.get("trace") or [])
        trace.append({"node": node, **extra})
        return trace

    def _analysed(
        self,
        state: AgentState,
        node: str,
        facts: list[str],
        graph_payload: dict | None = None,
        ui_action: dict | None = None,
        history: str = "",
        navigation_nodes: list[str] | None = None,
    ) -> dict:
        """Standard return for an analysis node."""
        payload: dict = {
            "facts": facts,
            "trace": self._trace(
                state, node, {"facts": len(facts), "history": bool(history)}
            ),
        }
        if graph_payload is not None:
            payload["graph_payload"] = graph_payload
        if ui_action is not None:
            payload["ui_action"] = ui_action
        if history:
            payload["history_text"] = history
        if navigation_nodes:
            payload["navigation_nodes"] = navigation_nodes
        return payload

    # -- entry point -------------------------------------------------------

    def ask(
        self,
        question: str,
        repository: str,
        conversation_id: str = "default",
        top_k: int = 5,
        explorer: dict | None = None,
    ) -> dict:
        """Run one turn. State persists per ``conversation_id``."""
        config = {"configurable": {"thread_id": f"{repository}:{conversation_id}"}}
        result = self.app.invoke(
            {
                "question": question,
                "repository": repository,
                "top_k": top_k,
                "explorer": explorer or {},
                # Reset per-turn outputs; focus and turns deliberately survive.
                "context": "",
                "sources": [],
                "chunks": [],
                "graph_context": None,
                "facts": [],
                "history_text": "",
                "navigation_nodes": [],
                "graph_payload": None,
                "ui_action": None,
                "navigation": None,
                "context_stats": {},
                "trace": [],
            },
            config=config,
        )
        return result
