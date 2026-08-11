"""Architecture, structure and dependency tools.

These answer the question class that measurement showed retrieval cannot: no
single chunk *is* a repository's architecture, so asking the vector store for
one returns whichever tutorial preamble scored best. The graph already knows
the module layout and the import edges, so these tools read it directly and
hand the model a factual skeleton instead of a guess.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.graph.model import NODE_FILE
from app.services.graph_service import GraphNotFoundError, GraphService

logger = logging.getLogger(__name__)


def _stamp(value: int | None) -> str:
    if not value:
        return "unknown"
    return datetime.fromtimestamp(int(value), tz=timezone.utc).strftime("%Y-%m-%d")


class ArchitectureTool:
    """Repository-wide structural summaries."""

    name = "analyse_architecture"

    def __init__(self, graph_service: GraphService | None = None):
        self.graph = graph_service or GraphService()

    # -- architecture ------------------------------------------------------

    def overview(self, repository: str) -> dict:
        return self.graph.overview(repository)

    def describe_architecture(self, repository: str) -> str:
        """A factual description of modules and their dependencies.

        Rendered as text because it becomes prompt context: the model needs
        the skeleton stated plainly, then explains it.
        """
        overview = self.graph.overview(repository)
        modules = [n for n in overview["nodes"] if n["kind"] == "module"]
        externals = [n for n in overview["nodes"] if n["kind"] == "external"]

        lines = [f"Repository: {repository}"]
        counts = overview["counts"]
        lines.append(
            f"Indexed: {counts.get('file', 0)} files, {counts.get('class', 0)} classes, "
            f"{counts.get('function', 0) + counts.get('method', 0)} functions/methods."
        )

        lines.append("")
        lines.append("Modules (largest first):")
        for module in sorted(modules, key=lambda m: -(m["data"].get("file_count") or 0)):
            data = module["data"]
            lines.append(
                f"  - {module['name']}: {data.get('file_count', 0)} files, "
                f"{data.get('line_count', 0)} lines, mainly {data.get('primary_language') or 'mixed'}"
            )

        # Never phrased as "first seen": history is walked to a commit ceiling,
        # so the earliest observed date is the start of the *window*, not of the
        # module. FastAPI's own history would otherwise be reported as starting
        # in 2025, and the model would repeat that as fact.
        window = self.graph.store.commit_range(repository)
        if window.get("first_at"):
            lines.append("")
            lines.append(
                f"(Dates below are limited to the indexed history window: "
                f"{window['total']} commits from {_stamp(window['first_at'])} "
                f"to {_stamp(window['last_at'])}. Earlier history was not walked, "
                f"so these are not module creation dates.)"
            )

        by_id = {m["id"]: m["name"] for m in modules}
        if overview["edges"]:
            lines.append("")
            lines.append("Module dependencies (source imports target):")
            for edge in overview["edges"]:
                source, target = by_id.get(edge["source"]), by_id.get(edge["target"])
                if source and target:
                    lines.append(f"  - {source} -> {target}")

        if externals:
            top = sorted(externals, key=lambda n: -(n["data"].get("uses") or 0))[:12]
            lines.append("")
            lines.append(
                "Most used external packages: "
                + ", ".join(f"{n['name']} ({n['data'].get('uses')})" for n in top)
            )

        return "\n".join(lines)

    # -- structure ---------------------------------------------------------

    def describe_structure(self, repository: str, max_children: int = 12) -> str:
        """A shallow directory tree — two levels, never the whole repository."""
        overview = self.graph.overview(repository)
        modules = [n for n in overview["nodes"] if n["kind"] == "module"]

        lines = [f"{repository}/"]
        for module in sorted(modules, key=lambda m: -(m["data"].get("file_count") or 0)):
            lines.append(f"  {module['name']}/  ({module['data'].get('file_count', 0)} files)")
            expanded = self.graph.expand(module["id"], limit=max_children)
            for child in expanded["nodes"]:
                marker = "/" if child["kind"] != NODE_FILE else ""
                lines.append(f"    {child['name']}{marker}")
            if expanded["truncated"]:
                remaining = expanded["total_children"] - len(expanded["nodes"])
                lines.append(f"    ... {remaining} more")
        return "\n".join(lines)

    # -- dependencies ------------------------------------------------------

    def describe_dependencies(self, node_id: str) -> tuple[str, dict]:
        """Both directions for a node, as text plus the raw payload.

        The payload is what the frontend highlights; the text is what the
        model reads. Returning both keeps the graph and the answer consistent.
        """
        result = self.graph.dependencies(node_id)
        node = result["node"]

        lines = [f"{node['kind']} {node['name']} ({node['key']})"]

        if result["dependencies"]:
            lines.append("")
            lines.append(f"Depends on ({len(result['dependencies'])}):")
            for dep in result["dependencies"][:20]:
                lines.append(f"  -> {dep['name']}  [{dep['key']}]")
        else:
            lines.append("")
            lines.append("Depends on: nothing inside this repository.")

        if result["dependents"]:
            lines.append("")
            lines.append(f"Used by ({len(result['dependents'])}):")
            for dep in result["dependents"][:20]:
                lines.append(f"  <- {dep['name']}  [{dep['key']}]")
        else:
            lines.append("")
            lines.append("Used by: nothing inside this repository.")

        return "\n".join(lines), result

    # -- history -----------------------------------------------------------

    def describe_history(self, repository: str, node_id: str | None = None) -> str:
        """Recent activity, for the whole repository or one file."""
        if node_id:
            try:
                commits = self.graph.history(node_id, limit=15)
                node = self.graph.store.get_node(node_id)
                label = node["name"] if node else node_id
            except GraphNotFoundError:
                return f"No history available for {node_id}."
            if not commits:
                return f"No recorded commits touch {label} within the indexed history window."
            lines = [f"Recent commits touching {label}:"]
            for commit in commits:
                lines.append(
                    f"  {_stamp(commit['authored_at'])}  {commit['sha'][:8]}  "
                    f"{commit['author']}: {commit['summary']}"
                )
            return "\n".join(lines)

        timeline = self.graph.timeline(repository)
        span = timeline["range"]
        lines = [
            f"Indexed history window for {repository}: {span.get('total', 0)} commits "
            f"from {_stamp(span.get('first_at'))} to {_stamp(span.get('last_at'))}. "
            f"Older commits were not walked, so dates are window-relative, not "
            f"creation dates.",
            "",
            "Activity by module:",
        ]
        for lane in timeline["lanes"][:12]:
            lines.append(
                f"  - {lane['module']}: {lane['total_commits']} commits, "
                f"earliest observed {_stamp(lane['first_seen'])}"
            )
        if timeline["releases"]:
            recent = timeline["releases"][-8:]
            lines.append("")
            lines.append(
                "Recent releases: "
                + ", ".join(f"{r['name']} ({_stamp(r['created_at'])})" for r in recent)
            )
        return "\n".join(lines)
