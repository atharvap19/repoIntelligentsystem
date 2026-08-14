"""Graph retrieval — the third source alongside dense and BM25.

Phase 2 treated the graph as a thing the agent *routed to*: a dependency
question went to the graph, everything else went to retrieval. That loses the
case the brief is built around. Asked "How does APIRouter connect to FastAPI?",
dense finds ``routing.py`` and ``applications.py``, BM25 finds the literal
``APIRouter`` and ``include_router``, and only the graph knows that
``routing.py`` is *imported by* ``applications.py``. Each source holds a third
of the answer, so all three run and are fused.

What this module contributes is a **neighbourhood**, not a graph. Seeds come
from the question's mentions, the Explorer's selection and the retrieved
chunks' own paths; the neighbourhood is then grown outward by import and
containment edges under a hard budget. FastAPI's graph has 6,363 nodes and
20k+ edges — sending it to a model is neither possible nor useful, so the
budget is a design constraint rather than a safety valve (Part 18).

Nothing here reads SQLite directly: every lookup goes through
``GraphService``, so a real graph database can replace the store without
touching this file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.graph.model import (
    EDGE_DEPENDS_ON,
    EDGE_IMPORTS,
    NODE_CLASS,
    NODE_DIRECTORY,
    NODE_EXTERNAL,
    NODE_FILE,
    NODE_FUNCTION,
    NODE_METHOD,
    NODE_MODULE,
    NODE_REPOSITORY,
)
from app.services.graph_service import GraphNotFoundError, GraphService

logger = logging.getLogger(__name__)

#: Nodes carried into the LLM context. Twenty-four fits comfortably beside the
#: code excerpts; beyond that the neighbourhood stops being a neighbourhood.
DEFAULT_NODE_BUDGET = 24

#: Edges rendered as facts. Kept below the node budget on purpose — a dense
#: subgraph is better summarised by its strongest edges than listed in full.
DEFAULT_EDGE_BUDGET = 40

#: How far to walk from a seed. One hop is "what touches this"; two reaches
#: "what touches the things that touch this", which is already the limit of
#: what a reader can hold. Three explodes on any real repository.
DEFAULT_HOPS = 2

#: Seeds resolved from question mentions. More than this and the question was
#: not really about specific symbols.
MAX_SEEDS = 6

#: Neighbours taken from one node, per direction. Hub modules have hundreds of
#: importers and enumerating them is neither possible within the budget nor
#: useful — see ``_neighbours``.
MAX_FANOUT = 6

SYMBOL_KINDS = frozenset({NODE_CLASS, NODE_FUNCTION, NODE_METHOD})
CONTAINER_KINDS = frozenset({NODE_REPOSITORY, NODE_MODULE, NODE_DIRECTORY})

#: Traversed by ``expand``. CONTAINS is deliberately excluded from the walk:
#: following containment from a module reaches every file beneath it and the
#: budget is spent before a single dependency is seen. Containment is added
#: back selectively, as a parent lookup, in ``_with_containers``.
WALK_EDGES = (EDGE_IMPORTS, EDGE_DEPENDS_ON)


def _implementation_prior(path: str) -> float:
    """How likely a path is to *define* something rather than use it.

    Delegates to the reranker's path priors so the graph and the text ranker
    agree on what counts as implementation. They must: a chunk demoted for
    living in ``docs_src`` should not be promoted back because the graph
    walked into the same file.
    """
    from app.rag.reranker import path_prior

    return path_prior(path or "")


@dataclass
class GraphHit:
    """One node in the retrieved neighbourhood, with why it is here."""

    node: dict
    #: 0 for a seed, 1 for its direct neighbours, and so on.
    distance: int
    #: ``mention`` | ``explorer`` | ``retrieval`` | ``imports`` | ``imported-by``
    #: | ``contains`` — what put this node in the neighbourhood.
    reason: str
    #: Seed this node was reached from, for explaining a path.
    via: str = ""

    @property
    def id(self) -> str:
        return self.node["id"]

    @property
    def kind(self) -> str:
        return self.node["kind"]

    @property
    def name(self) -> str:
        return self.node["name"]


@dataclass
class GraphContext:
    """A bounded slice of the repository graph, ready to become prompt text."""

    repository: str = ""
    hits: list[GraphHit] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    #: Seed node IDs, in the order they were resolved. The first is what the
    #: answer is "about" and what navigation centres on.
    seeds: list[str] = field(default_factory=list)
    #: Nodes that exist but were dropped by the budget, for honesty in the
    #: rendered text ("... and 31 more").
    withheld: int = 0

    def __bool__(self) -> bool:
        return bool(self.hits)

    @property
    def node_ids(self) -> set[str]:
        return {hit.id for hit in self.hits}

    def by_id(self, node_id: str) -> GraphHit | None:
        for hit in self.hits:
            if hit.id == node_id:
                return hit
        return None

    def paths(self) -> list[str]:
        """Relative paths present in the neighbourhood.

        Used to bias chunk ranking toward files the graph considers relevant —
        the join between structural and textual evidence.
        """
        found: list[str] = []
        for hit in self.hits:
            path = (
                hit.node["key"]
                if hit.kind == NODE_FILE
                else hit.node["data"].get("relative_path", "")
            )
            if path and path not in found:
                found.append(path)
        return found


class GraphRetriever:
    """Retrieves a bounded neighbourhood around whatever a question is about."""

    def __init__(
        self,
        graph_service: GraphService | None = None,
        node_budget: int = DEFAULT_NODE_BUDGET,
        edge_budget: int = DEFAULT_EDGE_BUDGET,
        hops: int = DEFAULT_HOPS,
    ):
        self.graph = graph_service or GraphService()
        self.node_budget = node_budget
        self.edge_budget = edge_budget
        self.hops = hops

    # -- entry point -------------------------------------------------------

    def retrieve(
        self,
        repository: str,
        mentions: list[str] | None = None,
        focus_node_id: str | None = None,
        explorer_node_ids: list[str] | None = None,
        chunk_paths: list[str] | None = None,
        hops: int | None = None,
    ) -> GraphContext:
        """Grow a neighbourhood from every available seed.

        Seeds are tried in priority order — an explicitly named symbol beats
        the Explorer's selection, which beats a path that merely happened to
        be retrieved — because the first seed becomes the navigation target.

        Never raises: a repository without a graph yields an empty context and
        the caller falls back to text retrieval alone.
        """
        context = GraphContext(repository=repository)
        if not repository:
            return context

        try:
            seeds = self._seeds(
                repository, mentions or [], focus_node_id, explorer_node_ids or [],
                chunk_paths or [],
            )
        except GraphNotFoundError:
            return context

        if not seeds:
            return context

        context.seeds = [hit.id for hit in seeds]
        # Lift before walking, not after: see :meth:`_with_containers`.
        frontier = seeds + self._with_containers(seeds, context)
        self._walk(frontier, seeds, context, hops if hops is not None else self.hops)
        self._collect_edges(context)

        logger.info(
            "graph retrieval: %d seeds -> %d nodes, %d edges (%d withheld)",
            len(seeds), len(context.hits), len(context.edges), context.withheld,
        )
        return context

    # -- seeding -----------------------------------------------------------

    def _seeds(
        self,
        repository: str,
        mentions: list[str],
        focus_node_id: str | None,
        explorer_node_ids: list[str],
        chunk_paths: list[str],
    ) -> list[GraphHit]:
        """Resolve every hint about the subject onto real graph nodes."""
        seeds: list[GraphHit] = []
        seen: set[str] = set()

        def add(node: dict | None, reason: str) -> None:
            if node and node["id"] not in seen and len(seeds) < MAX_SEEDS:
                seen.add(node["id"])
                seeds.append(GraphHit(node=node, distance=0, reason=reason))

        # 1. Explicitly named in the question — the strongest signal there is.
        for mention in mentions:
            add(self._resolve(repository, mention), "mention")

        # 2. The conversation's carried focus, or the Explorer's selection.
        if focus_node_id:
            add(self.graph.store.get_node(focus_node_id), "explorer")
        for node_id in explorer_node_ids:
            add(self.graph.store.get_node(node_id), "explorer")

        # 3. Nothing named and nothing selected: fall back to whatever text
        #    retrieval surfaced, so a vague question still gets structure.
        if not seeds:
            for path in chunk_paths[:MAX_SEEDS]:
                add(self.graph.store.find_node(repository, path, NODE_FILE), "retrieval")

        return seeds

    def _resolve(self, repository: str, term: str) -> dict | None:
        """Best graph node for a name written in a question.

        Exact-name lookup first, substring search only as a fallback. That
        order is not a micro-optimisation — it is the difference between right
        and wrong on a real repository. Substring search for ``FastAPI``
        matches every one of the 956 files under ``fastapi/`` through their
        path, and the shortest of those wins on tie-break, so the class
        actually named ``FastAPI`` is never reached.

        Within either result set, implementation outranks tests and examples.
        In FastAPI the library is under 5% of indexed files, so a name that
        appears in both is far more likely to be found in a tutorial first.
        """
        term = term.strip()
        if not term:
            return None

        try:
            exact = self.graph.store.find_by_name(repository, term)
            # Also try the bare stem, so "routing" reaches "routing.py".
            if not exact and "." not in term:
                exact = [
                    node
                    for suffix in (".py", ".ts", ".js", ".tsx", ".go", ".rs", ".java")
                    for node in self.graph.store.find_by_name(repository, term + suffix)
                ]
            candidates = exact or self.graph.search(repository, term, limit=40)
        except GraphNotFoundError:
            return None
        if not candidates:
            return None

        lowered = term.lower()
        stem = lowered.rsplit(".", 1)[0]

        def rank(node: dict) -> tuple:
            name = node["name"].lower()
            path = node["data"].get("relative_path") or node["key"]
            return (
                0 if name == lowered else 1 if name.rsplit(".", 1)[0] == stem else
                2 if name.startswith(lowered) else 3,
                # A tutorial that uses the name is not the thing named.
                -_implementation_prior(path),
                # Files and symbols are answerable; containers rarely are what
                # someone means when they name something.
                0 if node["kind"] == NODE_FILE else
                1 if node["kind"] in SYMBOL_KINDS else 2,
                len(name),
            )

        return sorted(candidates, key=rank)[0]

    # -- traversal ---------------------------------------------------------

    def _walk(
        self,
        frontier: list[GraphHit],
        seeds: list[GraphHit],
        context: GraphContext,
        hops: int,
    ) -> None:
        """Breadth-first expansion, stopping at the node budget.

        Breadth-first rather than depth-first so the budget buys the *closest*
        nodes. A depth-first walk would spend it all on one long import chain
        and never report what else touches the seed.
        """
        found: dict[str, GraphHit] = {hit.id: hit for hit in frontier}
        for hit in frontier:
            if hit not in context.hits:
                context.hits.append(hit)
        del seeds  # only the frontier drives traversal

        frontier = list(frontier)
        for distance in range(1, hops + 1):
            if len(context.hits) >= self.node_budget:
                # The budget is already spent, but silently returning here
                # would render a neighbourhood that looks complete. Count what
                # is being dropped so the rendered block can say so.
                context.withheld += sum(
                    1
                    for hit in frontier
                    for neighbour, _ in self._neighbours(hit.id)
                    if neighbour["id"] not in found
                )
                break

            next_frontier: list[GraphHit] = []
            for hit in frontier:
                for neighbour, reason in self._neighbours(hit.id):
                    if neighbour["id"] in found:
                        continue
                    if len(context.hits) >= self.node_budget:
                        context.withheld += 1
                        continue
                    entry = GraphHit(
                        node=neighbour,
                        distance=distance,
                        reason=reason,
                        via=hit.name,
                    )
                    found[neighbour["id"]] = entry
                    context.hits.append(entry)
                    next_frontier.append(entry)
            frontier = next_frontier

    def _neighbours(self, node_id: str) -> list[tuple[dict, str]]:
        """Both directions of the dependency edges around one node.

        Both, always: "what depends on this" and "what does this depend on"
        are the same question asked from opposite ends, and a reader following
        an answer needs the shape, not one arrow.

        Fan-out is capped and ranked, which measurement forced. FastAPI's
        ``fastapi/__init__.py`` has 580 importers; expanding it unranked fills
        the entire node budget with files named ``test_tutorial008d.py`` and
        the answer never reaches ``applications.py``. Listing 20 arbitrary
        importers of a hub says nothing anyway — the *count* is the fact, and
        the dependency tool reports that separately.
        """
        store = self.graph.store

        outgoing = store.outgoing(node_id, kinds=list(WALK_EDGES))
        incoming = store.incoming(node_id, kinds=list(WALK_EDGES))

        ids = [edge["target"] for edge in outgoing] + [edge["source"] for edge in incoming]
        if not ids:
            return []

        by_id = {node["id"]: node for node in store.nodes_by_ids(ids)}

        def collect(edges: list[dict], side: str, reason: str) -> list[tuple[dict, str]]:
            found = [
                (by_id[edge[side]], reason) for edge in edges if edge[side] in by_id
            ]
            # Implementation first, then larger files: between two tutorials
            # neither is informative, but between a tutorial and the module it
            # imports, the module is the answer.
            found.sort(
                key=lambda pair: (
                    -_implementation_prior(
                        pair[0]["data"].get("relative_path") or pair[0]["key"]
                    ),
                    -int(pair[0]["data"].get("line_count") or 0),
                )
            )
            return found[:MAX_FANOUT]

        return collect(outgoing, "target", "imports") + collect(
            incoming, "source", "imported-by"
        )

    def _with_containers(
        self, seeds: list[GraphHit], context: GraphContext
    ) -> list[GraphHit]:
        """Lift each symbol seed to the file that defines it.

        Two jobs, and the second is the important one.

        The obvious job: a bare ``APIRouter`` node tells the model nothing
        about where it lives, so its file and module are attached as an
        address. This is why CONTAINS is not walked generally but *is*
        followed upward here — one parent chain per seed is cheap.

        The job that makes the brief's own example work: **import edges are
        recorded between files, not between symbols.** Seeding on the classes
        ``APIRouter`` and ``FastAPI`` and walking IMPORTS from them therefore
        finds nothing at all — measured against the real FastAPI graph, that
        query returned five nodes and zero edges. Walking from their files
        finds the ``applications.py -> routing.py`` edge that is the actual
        answer. So the containing files are returned to the caller and join
        the traversal frontier, while the symbols stay the navigation seeds.
        """
        store = self.graph.store
        existing = context.node_ids
        lifted: list[GraphHit] = []

        for hit in seeds:
            if hit.kind in CONTAINER_KINDS:
                continue

            parent_id = hit.node.get("parent_id")
            while parent_id and len(context.hits) + len(lifted) < self.node_budget:
                if parent_id in existing:
                    break
                parent = store.get_node(parent_id)
                if parent is None:
                    break
                existing.add(parent_id)
                # Only the file carries import edges; modules and directories
                # are addresses, so they are recorded but not expanded from.
                is_file = parent["kind"] == NODE_FILE
                entry = GraphHit(
                    node=parent,
                    distance=1,
                    # "defines" rather than "contains": the file that defines
                    # the named symbol is part of the subject, and the
                    # renderer states its relationships rather than listing it
                    # as an incidental neighbour.
                    reason="defines" if is_file else "contains",
                    via=hit.name,
                )
                if is_file:
                    lifted.append(entry)
                else:
                    context.hits.append(entry)
                # Stop at the module: the repository root adds no information.
                if parent["kind"] in (NODE_MODULE, NODE_REPOSITORY):
                    break
                parent_id = parent.get("parent_id")

        return lifted

    def _collect_edges(self, context: GraphContext) -> None:
        """Every dependency edge whose *both* ends are in the neighbourhood.

        Edges to nodes the budget excluded are dropped rather than rendered as
        dangling references — a fact the model cannot check is worse than a
        fact it does not have.
        """
        ids = context.node_ids
        store = self.graph.store

        seen: set[str] = set()
        edges: list[dict] = []
        for hit in context.hits:
            if len(edges) >= self.edge_budget:
                break
            for edge in store.outgoing(hit.id, kinds=list(WALK_EDGES)):
                if edge["target"] in ids and edge["id"] not in seen:
                    seen.add(edge["id"])
                    edges.append(edge)
                    if len(edges) >= self.edge_budget:
                        break
        context.edges = edges


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


#: How a node kind reads in prompt text.
_KIND_LABEL = {
    NODE_REPOSITORY: "repository",
    NODE_MODULE: "module",
    NODE_DIRECTORY: "directory",
    NODE_FILE: "file",
    NODE_CLASS: "class",
    NODE_FUNCTION: "function",
    NODE_METHOD: "method",
    NODE_EXTERNAL: "external package",
}


def render_graph_context(context: GraphContext, max_lines: int = 60) -> str:
    """Turn a neighbourhood into the "GRAPH" block of the prompt.

    Written as an indented relationship list rather than as JSON: the model
    reads this as prose, and JSON spends tokens on punctuation that carry no
    meaning here.
    """
    if not context:
        return ""

    lines: list[str] = []
    by_id = {hit.id: hit for hit in context.hits}

    seeds = [by_id[s] for s in context.seeds if s in by_id]
    # The files that define the named symbols are part of the subject, not
    # background: they are where the import edges live, so they are what has
    # relationships worth stating.
    defining = [hit for hit in context.hits if hit.reason == "defines"]
    subjects = seeds + [hit for hit in defining if hit.id not in {s.id for s in seeds}]

    if seeds:
        lines.append("Subject of the question, located in the repository graph:")
        for hit in seeds:
            label = _KIND_LABEL.get(hit.kind, hit.kind)
            location = hit.node["data"].get("relative_path") or hit.node["key"]
            lines.append(f"  {label} {hit.name}  [{location}]")

    outgoing: dict[str, list[str]] = {}
    incoming: dict[str, list[str]] = {}
    for edge in context.edges:
        source, target = by_id.get(edge["source"]), by_id.get(edge["target"])
        if not source or not target:
            continue
        outgoing.setdefault(source.id, []).append(target.name)
        incoming.setdefault(target.id, []).append(source.name)

    for hit in subjects:
        depends = outgoing.get(hit.id, [])
        used_by = incoming.get(hit.id, [])
        if not depends and not used_by:
            continue
        lines.append("")
        location = hit.node["data"].get("relative_path") or hit.node["key"]
        lines.append(f"{location}:")
        if depends:
            lines.append(f"  depends on -> {', '.join(dict.fromkeys(depends))}")
        if used_by:
            lines.append(f"  used by <- {', '.join(dict.fromkeys(used_by))}")

    # Everything else reachable, so the model can see the wider shape without
    # each node needing its own paragraph.
    stated = {hit.id for hit in subjects}
    others = [
        hit for hit in context.hits
        if hit.id not in stated and hit.reason not in ("contains", "defines")
    ]
    if others:
        lines.append("")
        lines.append("Also in this neighbourhood:")
        for hit in others[: max_lines - len(lines)]:
            label = _KIND_LABEL.get(hit.kind, hit.kind)
            # The path, not just the name: a repository has many files called
            # `utils.py`, and two lines reading "file utils.py" are worse than
            # useless — the model cannot tell them apart.
            location = hit.node["data"].get("relative_path") or hit.node["key"]
            lines.append(f"  {label} {location} ({hit.reason} {hit.via})")

    if context.withheld:
        lines.append("")
        lines.append(
            f"({context.withheld} further related nodes were not included - the "
            f"neighbourhood is capped at {DEFAULT_NODE_BUDGET} nodes.)"
        )

    return "\n".join(lines)
