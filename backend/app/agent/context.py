"""Context building — the layer between retrieval and the model.

Part 15 of the brief asks for a pipeline rather than a concatenation, and the
distinction matters more than it sounds. Three sources return overlapping
evidence about the same repository: dense retrieval returns chunks, BM25
returns chunks, and the graph returns nodes whose files are frequently the
same files. Pasting all of it into a prompt produces a context that is
largely repetition, spends its budget on the fifth variation of one function,
and leaves no room for the one file that answers the question.

So this module does four things retrieval cannot do for itself:

**Deduplicate.** The same chunk arrives from both text strategies; the same
file arrives from the graph and from retrieval.

**Cross-rank.** A chunk whose file the graph considers structurally relevant
is more likely to be the answer than one that merely scored well on words.
That signal only exists once the sources are seen together, which is here.

**Budget.** Every block has a ceiling, and the ceilings are enforced before
the prompt is built rather than hoped for afterwards.

**Render.** One ordered document with labelled sections, so the model can
tell a graph fact from a retrieved guess.

``RepositoryContext`` is a value object: build it, read it, render it. It does
no I/O, which is what makes the whole pipeline testable without Chroma,
SQLite or Ollama.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.rag.graph_retriever import GraphContext, render_graph_context

logger = logging.getLogger(__name__)

#: Code excerpts carried into the prompt. Above ~8 the model starts averaging
#: across them instead of reading them, and on this hardware every excerpt is
#: paid for twice — once in prompt tokens, once in the ~150s generation.
DEFAULT_CODE_BUDGET = 8

#: Conversation turns replayed. Enough for "it" and "that" to resolve; short
#: enough that an old turn cannot outweigh the current question.
DEFAULT_HISTORY_TURNS = 6

#: Characters kept from one excerpt. A chunk longer than this is a whole file
#: that the chunker could not split, and its tail is rarely the answer.
MAX_EXCERPT_CHARS = 2_400

#: Characters of a previous answer replayed. Answers are long; what a
#: follow-up needs from them is the subject, not the prose.
MAX_REPLAYED_ANSWER = 400


@dataclass
class ExplorerContext:
    """What the user currently has open in Explorer.

    Part 3: the user should be able to select ``APIRouter`` and ask "what
    depends on this?" without naming it again. That only works if the
    selection travels with the question, so it is a first-class input to the
    turn rather than something the backend tries to infer.

    Every field is optional — the Chat view sends none of them, and the same
    code path has to serve both.
    """

    repository: str = ""
    module: str = ""
    file: str = ""
    symbol: str = ""
    #: Graph node ID of the current selection. The most precise field, and the
    #: only one that needs no resolution.
    node_id: str = ""
    #: Nodes currently emphasised in the graph view.
    focus_node_ids: list[str] = field(default_factory=list)
    #: Timeline cursor, if the user is looking at a past state.
    commit_sha: str = ""
    timestamp: int | None = None

    def __bool__(self) -> bool:
        return bool(
            self.node_id or self.file or self.symbol or self.module
            or self.focus_node_ids or self.commit_sha
        )

    def seed_ids(self) -> list[str]:
        """Node IDs this context offers as graph seeds, most precise first."""
        ids: list[str] = []
        if self.node_id:
            ids.append(self.node_id)
        for node_id in self.focus_node_ids:
            if node_id not in ids:
                ids.append(node_id)
        return ids

    def render(self) -> str:
        """The "CURRENT EXPLORER" block."""
        if not self:
            return ""
        lines = ["The user is looking at this in the repository explorer:"]
        for label, value in (
            ("repository", self.repository),
            ("module", self.module),
            ("file", self.file),
            ("symbol", self.symbol),
        ):
            if value:
                lines.append(f"  current {label}: {value}")
        if self.commit_sha:
            lines.append(f"  viewing history at commit: {self.commit_sha[:8]}")
        lines.append(
            '  Treat "this", "it" and "here" in the question as referring to the '
            "most specific item above."
        )
        return "\n".join(lines)


@dataclass
class ConversationTurn:
    """One prior exchange, as the agent needs to replay it."""

    question: str
    answer: str = ""
    focus_label: str = ""


@dataclass
class RepositoryContext:
    """Everything one turn will show the model, already selected and ordered.

    The field list follows Part 15 directly. Order here is also render order,
    and render order is deliberate: the question and the Explorer selection
    come first because they frame everything below them; graph facts precede
    code excerpts because the facts are complete and the excerpts are a
    sample; conversation comes last because it is the weakest evidence.
    """

    query: str = ""
    repository: str = ""
    intent: str = ""

    conversation: list[ConversationTurn] = field(default_factory=list)
    explorer: ExplorerContext = field(default_factory=ExplorerContext)

    #: Chunks after fusion, dedup and cross-ranking. One list, not two: which
    #: strategy found a chunk is a property of the chunk, not a partition.
    code: list[dict] = field(default_factory=list)
    graph: GraphContext = field(default_factory=GraphContext)

    #: Free-text blocks from the structural tools — architecture skeletons,
    #: dependency listings, commit summaries. Pre-rendered by their tool
    #: because each has its own shape.
    facts: list[str] = field(default_factory=list)
    history: str = ""

    #: Nodes an analysis node computed as the answer's subject, in order.
    #:
    #: Some answers have a subject the question never names. "How does a login
    #: request flow through this repository?" is *about* a chain of files but
    #: mentions none of them, so seed resolution finds nothing and navigation
    #: has nowhere to point — even though a derived flow is the most visual
    #: answer this system produces. The node that computed the chain records
    #: it here and the planner uses it as the target.
    navigation_nodes: list[str] = field(default_factory=list)

    #: Counters for the trace, so a thin context can be diagnosed.
    stats: dict[str, int] = field(default_factory=dict)

    # -- accessors ---------------------------------------------------------

    @property
    def semantic_results(self) -> list[dict]:
        """Chunks dense retrieval contributed to."""
        return [c for c in self.code if c.get("vector_score", 0.0) > 0]

    @property
    def lexical_results(self) -> list[dict]:
        """Chunks BM25 contributed to."""
        return [c for c in self.code if c.get("keyword_score", 0.0) > 0]

    def is_empty(self) -> bool:
        """True when there is nothing worth spending a generation on."""
        return not (self.code or self.graph or self.facts or self.history.strip())

    # -- rendering ---------------------------------------------------------

    def render(self) -> str:
        """Assemble the labelled context document.

        Section headers are shouted because the system prompt refers to them
        by name to set precedence — the model has to be able to tell which
        block a line came from.
        """
        blocks: list[str] = []

        explorer = self.explorer.render()
        if explorer:
            blocks.append(f"CURRENT EXPLORER CONTEXT\n{explorer}")

        if self.conversation:
            blocks.append(f"CONVERSATION SO FAR\n{self._render_conversation()}")

        graph_text = render_graph_context(self.graph)
        if graph_text:
            blocks.append(f"REPOSITORY GRAPH (from static analysis)\n{graph_text}")

        if self.facts:
            blocks.append(
                "REPOSITORY FACTS (computed over the whole repository)\n"
                + "\n\n".join(self.facts)
            )

        if self.history.strip():
            blocks.append(f"REPOSITORY HISTORY (from git)\n{self.history}")

        if self.code:
            blocks.append(f"RELEVANT CODE\n{self._render_code()}")

        return "\n\n---\n\n".join(blocks)

    def _render_conversation(self) -> str:
        lines: list[str] = []
        for turn in self.conversation:
            lines.append(f"User: {turn.question}")
            if turn.answer:
                answer = turn.answer.strip().replace("\n", " ")
                if len(answer) > MAX_REPLAYED_ANSWER:
                    answer = answer[:MAX_REPLAYED_ANSWER].rstrip() + " …"
                lines.append(f"Assistant: {answer}")
        return "\n".join(lines)

    def _render_code(self) -> str:
        """Excerpts with provenance, in the shape the prompt's rules expect."""
        from app.services.llm_service import describe_role

        blocks: list[str] = []
        for position, chunk in enumerate(self.code, start=1):
            path = chunk.get("relative_path", "unknown")
            header = f"[{position}] {path}"

            start, end = chunk.get("start_line"), chunk.get("end_line")
            if start and end:
                header += f":{start}-{end}"
            if chunk.get("symbol"):
                header += f"  symbol: {chunk['symbol']}"
            if chunk.get("chunk_type"):
                header += f"  kind: {chunk['chunk_type']}"
            header += f"  role: {describe_role(path, chunk.get('chunk_type', ''))}"
            if chunk.get("graph_linked"):
                # Says why this excerpt is here when its text score is weak.
                header += "  (also reached via the repository graph)"

            content = chunk.get("content", "")
            if len(content) > MAX_EXCERPT_CHARS:
                content = content[:MAX_EXCERPT_CHARS].rstrip() + "\n… (excerpt truncated)"

            blocks.append(f"{header}\n```\n{content}\n```")
        return "\n\n".join(blocks)


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------


def _chunk_key(chunk: dict) -> tuple:
    return (chunk.get("relative_path", ""), chunk.get("chunk_index", 0))


class ContextBuilder:
    """Assembles a ``RepositoryContext`` from raw source output.

    Pure functions over data the caller already fetched. It does not retrieve
    — the agent does that, because only the agent knows which sources a given
    intent needs — but everything after retrieval happens here.
    """

    def __init__(
        self,
        code_budget: int = DEFAULT_CODE_BUDGET,
        history_turns: int = DEFAULT_HISTORY_TURNS,
    ):
        self.code_budget = code_budget
        self.history_turns = history_turns

    def build(
        self,
        query: str,
        repository: str,
        intent: str = "",
        chunks: list[dict] | None = None,
        graph: GraphContext | None = None,
        explorer: ExplorerContext | None = None,
        conversation: list[ConversationTurn] | None = None,
        facts: list[str] | None = None,
        history: str = "",
        navigation_nodes: list[str] | None = None,
        code_budget: int | None = None,
    ) -> RepositoryContext:
        """Fuse, dedupe, rank and truncate every source into one context."""
        graph = graph or GraphContext()
        raw = chunks or []

        deduped = self._dedupe(raw)
        ranked = self._rank(deduped, graph)
        budget = code_budget if code_budget is not None else self.code_budget

        context = RepositoryContext(
            query=query,
            repository=repository,
            intent=intent,
            code=ranked[:budget],
            graph=graph,
            explorer=explorer or ExplorerContext(),
            conversation=(conversation or [])[-self.history_turns :],
            facts=[f for f in (facts or []) if f and f.strip()],
            history=history,
            navigation_nodes=list(navigation_nodes or []),
        )
        context.stats = {
            "chunks_in": len(raw),
            "chunks_deduped": len(deduped),
            "chunks_used": len(context.code),
            "graph_nodes": len(graph.hits),
            "graph_edges": len(graph.edges),
            "conversation_turns": len(context.conversation),
            "fact_blocks": len(context.facts),
        }
        logger.info("context built: %s", context.stats)
        return context

    # -- stages ------------------------------------------------------------

    @staticmethod
    def _dedupe(chunks: list[dict]) -> list[dict]:
        """Collapse repeats, keeping the best evidence from each appearance.

        The same chunk can arrive twice with different scores — once from a
        per-intent retrieval call, once from a broader one. Keeping the higher
        score of each channel means a chunk is never penalised for having been
        found twice.
        """
        merged: dict[tuple, dict] = {}
        for chunk in chunks:
            key = _chunk_key(chunk)
            existing = merged.get(key)
            if existing is None:
                merged[key] = dict(chunk)
                continue
            for channel in ("score", "vector_score", "keyword_score", "rerank_score"):
                existing[channel] = max(
                    float(existing.get(channel, 0.0) or 0.0),
                    float(chunk.get(channel, 0.0) or 0.0),
                )
        return list(merged.values())

    @staticmethod
    def _rank(chunks: list[dict], graph: GraphContext) -> list[dict]:
        """Re-order chunks using structural evidence retrieval could not see.

        A chunk from a file the graph placed in the neighbourhood is promoted;
        a chunk from the *seed* file is promoted harder. This is the fusion
        the brief is really asking for — not vector and BM25 (the hybrid
        retriever already does that), but text evidence against graph
        evidence.

        The boost is multiplicative and bounded so it re-orders near-ties
        without letting structure overrule a decisively better text match.
        """
        if not chunks:
            return []

        neighbourhood = set(graph.paths())
        seed_paths = set()
        for seed_id in graph.seeds:
            hit = graph.by_id(seed_id)
            if hit is None:
                continue
            path = hit.node["key"] if hit.kind == "file" else hit.node["data"].get(
                "relative_path", ""
            )
            if path:
                seed_paths.add(path)

        for chunk in chunks:
            path = chunk.get("relative_path", "")
            base = float(chunk.get("score", 0.0) or 0.0)

            if path in seed_paths:
                boost, linked = 1.6, True
            elif path in neighbourhood:
                boost, linked = 1.25, True
            else:
                boost, linked = 1.0, False

            chunk["graph_linked"] = linked
            chunk["graph_boost"] = boost
            # Ranking is a separate key from `score` so the API keeps
            # reporting the retrieval score the Inspector explains.
            chunk["context_score"] = base * boost

        return sorted(chunks, key=lambda c: c["context_score"], reverse=True)
