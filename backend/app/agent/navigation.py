"""Structured navigation targets.

Part 6 is explicit that the frontend must not parse natural language to find
out where to navigate. That is the right call for a reason worth stating: the
answer text is model output, so any parser over it is a parser over something
that changes with temperature, model version and phrasing. A link that
silently stops working when the model rewords a sentence is worse than no
link. So navigation is decided by deterministic code from data the backend
already holds, and travels beside the answer rather than inside it.

Part 5 adds the other half: navigation is *offered*, never forced. "How many
Python files are in this repository?" has a correct answer and no meaningful
visual target, and putting an [Explore this] button under it trains the user
to ignore the button. So ``decide`` returns an unavailable target more often
than not, and says why.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from app.agent.intents import (
    INTENT_ARCHITECTURE,
    INTENT_CONCEPT,
    INTENT_DEPENDENCY,
    INTENT_FILE,
    INTENT_FLOW,
    INTENT_HISTORY,
    INTENT_HOTSPOT,
    INTENT_READONLY,
    INTENT_SEARCH,
    INTENT_STRUCTURE,
)
from app.agent.context import RepositoryContext
from app.graph.model import (
    NODE_CLASS,
    NODE_DIRECTORY,
    NODE_EXTERNAL,
    NODE_FILE,
    NODE_FUNCTION,
    NODE_METHOD,
    NODE_MODULE,
    NODE_REPOSITORY,
)

logger = logging.getLogger(__name__)

# -- target types (Part 6) -------------------------------------------------

TARGET_REPOSITORY = "repository"
TARGET_MODULE = "module"
TARGET_FOLDER = "folder"
TARGET_FILE = "file"
TARGET_CLASS = "class"
TARGET_FUNCTION = "function"
TARGET_SYMBOL = "symbol"
TARGET_NEIGHBOURHOOD = "graph_neighborhood"
TARGET_TIMELINE = "timeline"
TARGET_COMMIT = "commit"
TARGET_ARCHITECTURE = "architecture"

TARGET_TYPES: tuple[str, ...] = (
    TARGET_REPOSITORY,
    TARGET_MODULE,
    TARGET_FOLDER,
    TARGET_FILE,
    TARGET_CLASS,
    TARGET_FUNCTION,
    TARGET_SYMBOL,
    TARGET_NEIGHBOURHOOD,
    TARGET_TIMELINE,
    TARGET_COMMIT,
    TARGET_ARCHITECTURE,
)

#: Graph node kind -> navigation target type.
_KIND_TO_TARGET = {
    NODE_REPOSITORY: TARGET_REPOSITORY,
    NODE_MODULE: TARGET_MODULE,
    NODE_DIRECTORY: TARGET_FOLDER,
    NODE_FILE: TARGET_FILE,
    NODE_CLASS: TARGET_CLASS,
    NODE_FUNCTION: TARGET_FUNCTION,
    NODE_METHOD: TARGET_SYMBOL,
    NODE_EXTERNAL: TARGET_SYMBOL,
}

# -- focus modes -----------------------------------------------------------

#: Centre the node and select it. The default for a single subject.
FOCUS_CENTER = "center"
#: Centre it and highlight its dependency edges — "what depends on this".
FOCUS_NEIGHBOURHOOD = "neighborhood"
#: Expand the container and show its children.
FOCUS_EXPAND = "expand"
#: Move the timeline rather than the graph.
FOCUS_TIMELINE = "timeline"
#: Draw an ordered chain through the graph — a derived flow.
FOCUS_FLOW = "flow"


@dataclass
class NavigationTarget:
    """Where the frontend should take the user, if anywhere.

    Serialised straight onto the API response. ``available`` is the only field
    a client must check; everything else is meaningful only when it is true.
    """

    available: bool = False
    target_type: str = ""
    repository: str = ""
    module: str = ""
    file: str = ""
    symbol: str = ""
    #: Exact graph node to select. Present whenever the target resolved to a
    #: real node, which lets the frontend skip a lookup.
    node_id: str = ""
    #: Additional nodes worth showing — the neighbourhood, or a flow's steps.
    related_node_ids: list[str] = field(default_factory=list)
    focus_mode: str = FOCUS_CENTER
    #: Button text. Built from real names so it reads as a place, not a verb.
    label: str = ""
    #: Timeline / commit targets only.
    commit_sha: str = ""
    #: Why there is nothing to explore. Diagnostic, never shown as an error.
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _unavailable(reason: str) -> NavigationTarget:
    return NavigationTarget(available=False, reason=reason)


class NavigationPlanner:
    """Decides the exploration target for an answered turn."""

    def decide(self, context: RepositoryContext) -> NavigationTarget:
        """Pick a target from the context the answer was built on.

        Reads only ``context`` — never the answer text — so the decision is
        reproducible and testable without a model.
        """
        intent = context.intent
        repository = context.repository

        if intent == INTENT_READONLY:
            # Nothing to explore for a request the product declines to do.
            return _unavailable("read-only refusal has no exploration target")

        if intent in (INTENT_ARCHITECTURE, INTENT_STRUCTURE):
            return NavigationTarget(
                available=True,
                target_type=TARGET_ARCHITECTURE if intent == INTENT_ARCHITECTURE
                else TARGET_REPOSITORY,
                repository=repository,
                focus_mode=FOCUS_EXPAND,
                label=f"Explore {repository}"
                if intent == INTENT_STRUCTURE
                else "Explore the architecture",
            )

        if intent == INTENT_HISTORY:
            return self._history_target(context)

        # Answers whose subject was *computed* rather than named. A flow
        # question mentions no file, and "which files are most depended on?"
        # names nothing either, so seed resolution finds nothing to point at —
        # yet both produce a concrete set of nodes, and the flow in particular
        # is the most visual answer this system gives. The analysis node hands
        # its nodes over in `navigation_nodes`.
        if context.navigation_nodes:
            return self._computed_target(context, intent)

        seed = self._primary_seed(context)

        # A seed reached only through retrieval is incidental: nobody named it
        # and nobody selected it, it is just where the best chunk happened to
        # live. "How many Python files are there?" resolves such a seed on
        # every run, and offering to explore it would put a button under an
        # answer that has no place to go. Those fall through to the same
        # intent gate as a context with no seed at all.
        if seed is None or seed.reason == "retrieval":
            return self._from_code(context)

        return self._from_seed(context, seed, intent)

    # -- branches ----------------------------------------------------------

    @staticmethod
    def _primary_seed(context: RepositoryContext):
        """The node the answer is about, if the graph resolved one."""
        for seed_id in context.graph.seeds:
            hit = context.graph.by_id(seed_id)
            if hit is not None:
                return hit
        return None

    def _from_seed(self, context: RepositoryContext, seed, intent: str) -> NavigationTarget:
        node = seed.node
        data = node.get("data", {})
        target_type = _KIND_TO_TARGET.get(seed.kind, TARGET_SYMBOL)

        is_file = seed.kind == NODE_FILE
        file_path = node["key"] if is_file else data.get("relative_path", "")

        # Dependency, concept and flow questions are *about* the shape around
        # the node, so they open the neighbourhood rather than the node alone.
        wants_neighbourhood = intent in (INTENT_DEPENDENCY, INTENT_FLOW, INTENT_CONCEPT)

        related = [
            hit.id for hit in context.graph.hits
            if hit.id != seed.id and hit.reason in ("imports", "imported-by")
        ]

        if wants_neighbourhood and related:
            focus_mode = FOCUS_NEIGHBOURHOOD
            target_type = TARGET_NEIGHBOURHOOD if intent != INTENT_DEPENDENCY else target_type
        elif seed.kind in (NODE_MODULE, NODE_DIRECTORY):
            focus_mode = FOCUS_EXPAND
        else:
            focus_mode = FOCUS_CENTER

        return NavigationTarget(
            available=True,
            target_type=target_type,
            repository=context.repository,
            module=data.get("module", "") or (node["key"] if seed.kind == NODE_MODULE else ""),
            file=file_path,
            symbol="" if is_file or seed.kind in (NODE_MODULE, NODE_DIRECTORY) else seed.name,
            node_id=seed.id,
            related_node_ids=related[:20],
            focus_mode=focus_mode,
            label=self._label(seed, focus_mode),
        )

    def _computed_target(self, context: RepositoryContext, intent: str) -> NavigationTarget:
        """Point at nodes an analysis node derived, not ones the user named."""
        nodes = context.navigation_nodes
        head = nodes[0]

        if intent == INTENT_FLOW:
            return NavigationTarget(
                available=True,
                target_type=TARGET_NEIGHBOURHOOD,
                repository=context.repository,
                node_id=head,
                # Order matters here in a way it does not elsewhere: the graph
                # numbers these as steps, so the chain must arrive intact.
                related_node_ids=nodes[1:],
                focus_mode=FOCUS_FLOW,
                label="Explore this flow",
            )

        return NavigationTarget(
            available=True,
            target_type=TARGET_NEIGHBOURHOOD,
            repository=context.repository,
            node_id=head,
            related_node_ids=nodes[1:],
            focus_mode=FOCUS_NEIGHBOURHOOD,
            label="Explore these files",
        )

    def _from_code(self, context: RepositoryContext) -> NavigationTarget:
        """Fall back to the best-ranked retrieved file.

        Only when retrieval was confident and the question was a locating one
        — "where is authentication implemented?" — because offering to
        explore an incidentally-retrieved file is noise.
        """
        if context.intent not in (INTENT_SEARCH, INTENT_FILE, INTENT_CONCEPT):
            return _unavailable("no structural subject and intent is not locational")
        if not context.code:
            return _unavailable("no retrieved code to point at")

        best = context.code[0]
        path = best.get("relative_path", "")
        if not path:
            return _unavailable("top chunk has no path")

        return NavigationTarget(
            available=True,
            target_type=TARGET_FILE,
            repository=context.repository,
            file=path,
            symbol=best.get("symbol", "") or "",
            # Node IDs are deterministic, so this is constructible without a
            # lookup — see app.graph.model.node_id.
            node_id=f"{context.repository}::{NODE_FILE}::{path}",
            focus_mode=FOCUS_CENTER,
            label=f"Explore {path.split('/')[-1]}",
        )

    def _history_target(self, context: RepositoryContext) -> NavigationTarget:
        """History questions navigate the timeline, not the graph."""
        seed = self._primary_seed(context)
        if seed is not None:
            data = seed.node.get("data", {})
            return NavigationTarget(
                available=True,
                target_type=TARGET_TIMELINE,
                repository=context.repository,
                module=data.get("module", ""),
                file=seed.node["key"] if seed.kind == NODE_FILE
                else data.get("relative_path", ""),
                node_id=seed.id,
                focus_mode=FOCUS_TIMELINE,
                label=f"See {seed.name} on the timeline",
            )
        return NavigationTarget(
            available=True,
            target_type=TARGET_TIMELINE,
            repository=context.repository,
            focus_mode=FOCUS_TIMELINE,
            label="Open the timeline",
        )

    @staticmethod
    def _label(seed, focus_mode: str) -> str:
        """Button text naming a real place in the repository."""
        if focus_mode == FOCUS_NEIGHBOURHOOD:
            return f"Explore what connects to {seed.name}"
        if focus_mode == FOCUS_EXPAND:
            return f"Explore {seed.name}"
        return f"Explore {seed.name}"


def commit_target(repository: str, sha: str, summary: str = "") -> NavigationTarget:
    """Target for a specific commit, used by history answers that name one."""
    label = f"View commit {sha[:8]}"
    if summary:
        label = f"View {sha[:8]} — {summary[:48]}"
    return NavigationTarget(
        available=True,
        target_type=TARGET_COMMIT,
        repository=repository,
        commit_sha=sha,
        focus_mode=FOCUS_TIMELINE,
        label=label,
    )
