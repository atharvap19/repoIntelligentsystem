"""Computed repository insights: hotspots and flows.

Both answer question classes that neither retrieval nor a single graph lookup
can reach, and both are computed from data already in the store — Part 13 is
explicit that a metric must not be invented, so every number here is a count
of something real and every insight carries the count that produced it.

**Hotspots** rank nodes by three independent signals: how many things import
them (structural load), how often they change (churn), and how large they are
(complexity proxy). They are reported separately rather than blended into one
score, because "27 modules import this" and "this changed 40 times" are
different problems and a combined number would hide which one is present.

**Flows** are paths through the import graph. A repository's request path is
not written down anywhere, but it is implied by which layer imports which, so
a flow is derived by walking imports from an entry point and grouping what it
reaches into depth layers. This is honest about being static: it reports what
*can* call what, not what did.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.graph.model import (
    EDGE_DEPENDS_ON,
    EDGE_IMPORTS,
    NODE_FILE,
    NODE_MODULE,
)
from app.graph.store import GraphStore

logger = logging.getLogger(__name__)

#: Hotspots reported per category.
DEFAULT_HOTSPOT_LIMIT = 8

#: A node needs at least this many dependents before it is worth calling a
#: structural hotspot. Below it, the "hotspot" is just a file with a couple of
#: importers, and saying so is noise.
MIN_DEPENDENTS = 3

#: Likewise for churn: two commits is not a pattern.
MIN_CHANGES = 3

#: Depth of a derived flow. Beyond four layers the chain stops describing a
#: flow and starts describing the whole repository.
MAX_FLOW_DEPTH = 4

#: Files per flow layer.
MAX_FLOW_WIDTH = 6

#: Paths that conventionally start a request or a process. Ordered — the first
#: pattern that matches anything wins, so a real HTTP entry point beats a CLI.
_ENTRY_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"(^|/)(main|app|application|server|asgi|wsgi)\.(py|js|ts)$", re.I), "application entry point"),
    (re.compile(r"(^|/)(routes?|routing|urls|endpoints?|controllers?|api)(/|\.)", re.I), "routing layer"),
    (re.compile(r"(^|/)(views?|handlers?|resolvers?)(/|\.)", re.I), "request handlers"),
    (re.compile(r"(^|/)(cli|__main__|manage)\.(py|js|ts)$", re.I), "command line entry point"),
)


@dataclass
class Hotspot:
    """One node flagged as notable, with the measurement behind it."""

    node_id: str
    name: str
    kind: str
    path: str
    #: ``structural`` | ``churn`` | ``complexity``
    category: str
    #: The measured value — dependents, commits, or lines.
    value: int
    #: Plain-language justification, built from the value. Part 13 requires
    #: the "why", and building it here means the model never has to guess it.
    reason: str
    #: Names of a few actual dependents, so the claim is checkable.
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "kind": self.kind,
            "path": self.path,
            "category": self.category,
            "value": self.value,
            "reason": self.reason,
            "evidence": self.evidence,
        }


@dataclass
class FlowStep:
    """One layer of a derived flow."""

    depth: int
    label: str
    nodes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "depth": self.depth,
            "label": self.label,
            "nodes": [
                {"id": n["id"], "name": n["name"], "path": n["key"], "kind": n["kind"]}
                for n in self.nodes
            ],
        }


@dataclass
class Flow:
    """A static path through the repository, entry point first."""

    repository: str
    entry_label: str = ""
    steps: list[FlowStep] = field(default_factory=list)
    #: Node IDs in order, for the Explorer to highlight as a chain.
    node_ids: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.steps)

    def to_dict(self) -> dict:
        return {
            "repository": self.repository,
            "entry_label": self.entry_label,
            "steps": [s.to_dict() for s in self.steps],
            "node_ids": self.node_ids,
        }


class InsightAnalyzer:
    """Computes hotspots and flows from a stored graph."""

    def __init__(self, store: GraphStore):
        self.store = store

    # -- hotspots ----------------------------------------------------------

    def hotspots(self, repository: str, limit: int = DEFAULT_HOTSPOT_LIMIT) -> list[Hotspot]:
        """Structural, churn and complexity hotspots, best evidence first."""
        found: list[Hotspot] = []
        found.extend(self._structural(repository, limit))
        found.extend(self._churn(repository, limit))
        found.extend(self._complexity(repository, limit))
        return found

    def _structural(self, repository: str, limit: int) -> list[Hotspot]:
        """Nodes many other nodes import."""
        counts = self.store.dependent_counts(
            repository, kinds=[EDGE_IMPORTS, EDGE_DEPENDS_ON]
        )
        ranked = sorted(
            ((node_id, n) for node_id, n in counts.items() if n >= MIN_DEPENDENTS),
            key=lambda pair: -pair[1],
        )[:limit]
        if not ranked:
            return []

        nodes = {n["id"]: n for n in self.store.nodes_by_ids([r[0] for r in ranked])}

        hotspots: list[Hotspot] = []
        for node_id, count in ranked:
            node = nodes.get(node_id)
            if node is None or node["kind"] not in (NODE_FILE, NODE_MODULE):
                continue

            importers = self.store.incoming(node_id, kinds=[EDGE_IMPORTS, EDGE_DEPENDS_ON])
            names = [
                n["name"]
                for n in self.store.nodes_by_ids([e["source"] for e in importers[:5]])
            ]
            noun = "modules" if node["kind"] == NODE_MODULE else "files"
            hotspots.append(
                Hotspot(
                    node_id=node_id,
                    name=node["name"],
                    kind=node["kind"],
                    path=node["key"],
                    category="structural",
                    value=count,
                    reason=(
                        f"{count} {noun} in this repository import {node['name']}, "
                        f"so a change to it can affect all of them."
                    ),
                    evidence=names,
                )
            )
        return hotspots

    def _churn(self, repository: str, limit: int) -> list[Hotspot]:
        """Files that change most often within the indexed history window."""
        counts = self.store.change_counts(repository)
        ranked = sorted(
            ((path, n) for path, n in counts.items() if n >= MIN_CHANGES),
            key=lambda pair: -pair[1],
        )

        hotspots: list[Hotspot] = []
        for path, count in ranked:
            # Only report paths that are actually indexed. History covers
            # markdown, CI config and lock files the parser excludes, and
            # calling those "hotspots" answers a question nobody asked.
            node = self.store.find_node(repository, path, NODE_FILE)
            if node is None:
                continue
            hotspots.append(
                Hotspot(
                    node_id=node["id"],
                    name=node["name"],
                    kind=node["kind"],
                    path=path,
                    category="churn",
                    value=count,
                    reason=(
                        f"{path} was changed in {count} commits within the indexed "
                        f"history window — the most frequently modified indexed file."
                    ),
                )
            )
            if len(hotspots) >= limit:
                break
        return hotspots

    def _complexity(self, repository: str, limit: int) -> list[Hotspot]:
        """The largest files, as a stand-in for complexity.

        Line count is a weak proxy and is labelled as one. The analyser
        already computes a complexity band, so that is preferred when present
        and lines only break the tie.
        """
        files = self.store.nodes_of_kind(repository, NODE_FILE)
        ranked = sorted(
            files,
            key=lambda n: (
                0 if n["data"].get("complexity") == "high" else 1,
                -int(n["data"].get("line_count") or 0),
            ),
        )[:limit]

        return [
            Hotspot(
                node_id=node["id"],
                name=node["name"],
                kind=node["kind"],
                path=node["key"],
                category="complexity",
                value=int(node["data"].get("line_count") or 0),
                reason=(
                    f"{node['key']} is {node['data'].get('line_count', 0)} lines with "
                    f"{node['data'].get('class_count', 0)} classes and "
                    f"{node['data'].get('function_count', 0)} functions "
                    f"(complexity band: {node['data'].get('complexity', 'unknown')})."
                ),
            )
            for node in ranked
            if int(node["data"].get("line_count") or 0) > 0
        ]

    # -- flow --------------------------------------------------------------

    def flow(self, repository: str, start_node_id: str | None = None) -> Flow:
        """Derive a layered path through the import graph.

        With no starting point, an entry file is guessed from conventional
        naming (``main.py``, a ``routes/`` directory). That guess is reported
        as ``entry_label`` so the answer can say where it started rather than
        presenting the flow as discovered fact.
        """
        flow = Flow(repository=repository)

        start = (
            self.store.get_node(start_node_id)
            if start_node_id
            else self._guess_entry(repository)
        )
        if start is None:
            return flow

        flow.entry_label = self._entry_label(start["key"])

        seen = {start["id"]}
        layer = [start]
        flow.steps.append(FlowStep(depth=0, label=flow.entry_label, nodes=[start]))
        flow.node_ids.append(start["id"])

        for depth in range(1, MAX_FLOW_DEPTH):
            next_layer: list[dict] = []
            for node in layer:
                edges = self.store.outgoing(node["id"], kinds=[EDGE_IMPORTS])
                targets = self.store.nodes_by_ids([e["target"] for e in edges])
                for target in targets:
                    if target["id"] in seen or len(next_layer) >= MAX_FLOW_WIDTH:
                        continue
                    seen.add(target["id"])
                    next_layer.append(target)

            if not next_layer:
                break

            flow.steps.append(
                FlowStep(
                    depth=depth,
                    label=self._layer_label(next_layer),
                    nodes=next_layer,
                )
            )
            flow.node_ids.extend(n["id"] for n in next_layer)
            layer = next_layer

        return flow

    def _guess_entry(self, repository: str) -> dict | None:
        """The most entry-point-looking file in the repository."""
        files = self.store.nodes_of_kind(repository, NODE_FILE)
        if not files:
            return None

        for pattern, _ in _ENTRY_PATTERNS:
            matches = [f for f in files if pattern.search(f["key"])]
            if matches:
                # Shallowest path wins: `app/main.py` is an entry point,
                # `tests/fixtures/app/main.py` is a fixture.
                return min(matches, key=lambda f: (f["key"].count("/"), len(f["key"])))

        # Nothing conventional: fall back to the file with the most importers,
        # which is the closest thing to a hub this repository has.
        counts = self.store.dependent_counts(repository, kinds=[EDGE_IMPORTS])
        if counts:
            best = max(counts, key=counts.get)
            return self.store.get_node(best)
        return None

    @staticmethod
    def _entry_label(path: str) -> str:
        for pattern, label in _ENTRY_PATTERNS:
            if pattern.search(path):
                return label
        return "starting point"

    @staticmethod
    def _layer_label(nodes: list[dict]) -> str:
        """Name a layer after the directory its files share, when they share one."""
        modules = {n["data"].get("module", "") for n in nodes if n["data"].get("module")}
        if len(modules) == 1:
            return next(iter(modules))
        directories = {n["data"].get("directory", "") for n in nodes}
        if len(directories) == 1 and next(iter(directories)):
            return next(iter(directories))
        return "imports"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_hotspots(hotspots: list[Hotspot]) -> str:
    """Hotspots as prompt text, grouped so each category reads as a claim."""
    if not hotspots:
        return ""

    by_category: dict[str, list[Hotspot]] = {}
    for hotspot in hotspots:
        by_category.setdefault(hotspot.category, []).append(hotspot)

    headings = {
        "structural": "Most depended-on (counted from import edges)",
        "churn": "Most frequently changed (counted from git history)",
        "complexity": "Largest files (line counts from static analysis)",
    }

    lines: list[str] = []
    for category in ("structural", "churn", "complexity"):
        entries = by_category.get(category)
        if not entries:
            continue
        lines.append(headings[category] + ":")
        for hotspot in entries:
            lines.append(f"  - {hotspot.reason}")
            if hotspot.evidence:
                lines.append(f"      importers include: {', '.join(hotspot.evidence)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def render_flow(flow: Flow) -> str:
    """A flow as an arrow chain, which is how it will be drawn in Explorer."""
    if not flow:
        return ""

    lines = [
        f"Static call/import path derived from the import graph, starting at the "
        f"{flow.entry_label}:",
        "",
    ]
    # ASCII arrows deliberately: this text can reach a Windows console, which
    # is cp1252 here and raises UnicodeEncodeError on box-drawing characters.
    # The same constraint is why commit messages are never logged.
    for step in flow.steps:
        names = ", ".join(n["name"] for n in step.nodes)
        arrow = "" if step.depth == 0 else "   |\n   v\n"
        lines.append(f"{arrow}  [{step.label}] {names}")

    lines.append("")
    lines.append(
        "(This is what the import graph permits, not an observed runtime trace. "
        "Dynamic dispatch and runtime registration are not visible to static analysis.)"
    )
    return "\n".join(lines)
