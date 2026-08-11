"""Builds a repository graph from parsed files.

Consumes ``RepositoryParser`` output directly — the same filtered file stream
the chunker sees — so the graph and the vector index always describe the same
corpus. A file excluded from retrieval is excluded from the graph.

Module assignment is the one judgement call here. A "module" is the top-level
directory a file lives under, except that repositories which put everything in
a single source root (``src/``, ``lib/``, ``app/``) get their second level
instead, because otherwise every file lands in one module and the graph says
nothing.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from app.graph.analyzer import FileAnalysis, analyse
from app.graph.model import (
    EDGE_CONTAINS,
    EDGE_DEPENDS_ON,
    EDGE_EXTENDS,
    EDGE_IMPORTS,
    NODE_CLASS,
    NODE_DIRECTORY,
    NODE_EXTERNAL,
    NODE_FILE,
    NODE_FUNCTION,
    NODE_METHOD,
    NODE_MODULE,
    NODE_REPOSITORY,
    Edge,
    Node,
    RepositoryGraph,
    node_id,
)

logger = logging.getLogger(__name__)

#: Directories that are packaging conventions rather than modules. A file in
#: ``src/auth/`` belongs to ``auth``, not to ``src``.
SOURCE_ROOTS = frozenset({"src", "lib", "app", "source", "sources", "pkg", "internal"})

#: Import prefixes that are unambiguously external, skipping resolution work.
_RELATIVE_MARKERS = (".", "/")


@dataclass
class BuildStats:
    files: int = 0
    symbols: int = 0
    internal_imports: int = 0
    external_imports: int = 0
    unresolved_imports: int = 0
    parse_failures: int = 0
    seconds: float = 0.0
    languages: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.files} files, {self.symbols} symbols, "
            f"{self.internal_imports} internal / {self.external_imports} external imports, "
            f"{self.unresolved_imports} unresolved, {self.parse_failures} parse failures "
            f"in {self.seconds:.2f}s"
        )


def module_of(relative_path: str) -> str:
    """Which top-level area a file belongs to."""
    parts = [p for p in relative_path.split("/") if p]
    if len(parts) <= 1:
        return "(root)"
    head = parts[0]
    if head.lower() in SOURCE_ROOTS and len(parts) > 2:
        return f"{head}/{parts[1]}"
    return head


def directory_of(relative_path: str) -> str:
    parts = relative_path.split("/")
    return "/".join(parts[:-1]) if len(parts) > 1 else ""


class GraphBuilder:
    """Turns parsed files into nodes and edges."""

    def __init__(self, repository: str):
        self.repository = repository
        self.graph = RepositoryGraph(repository=repository)
        self.stats = BuildStats()

        self._nodes: dict[str, Node] = {}
        #: relative_path -> file node id, for import resolution.
        self._file_ids: dict[str, str] = {}
        #: Dotted module path -> relative_path, e.g. ``app.rag.parser``.
        self._module_index: dict[str, str] = {}
        self._analyses: dict[str, FileAnalysis] = {}

    # -- public ------------------------------------------------------------

    def build(self, parsed_files: Iterable[dict]) -> RepositoryGraph:
        """Two passes: structure first, then dependencies.

        Imports can only be resolved once every file is known, so nothing is
        resolved during the first pass.
        """
        import time

        started = time.perf_counter()

        for parsed in parsed_files:
            self._add_file(parsed)

        self._add_dependencies()

        self.stats.seconds = time.perf_counter() - started
        logger.info("Built graph for %r: %s", self.repository, self.stats.summary())
        return self.graph

    # -- structure ---------------------------------------------------------

    def _node(
        self, kind: str, key: str, name: str, parent_id: str | None, data: dict | None = None
    ) -> Node:
        """Create a node once; repeated calls return the existing one."""
        identifier = node_id(self.repository, kind, key)
        existing = self._nodes.get(identifier)
        if existing is not None:
            if data:
                existing.data.update(data)
            return existing

        node = Node(
            id=identifier,
            repository=self.repository,
            kind=kind,
            name=name,
            key=key,
            parent_id=parent_id,
            data=data or {},
        )
        self._nodes[identifier] = node
        self.graph.add_node(node)
        if parent_id:
            self.graph.add_edge(
                Edge(self.repository, EDGE_CONTAINS, parent_id, identifier)
            )
        return node

    def _repository_node(self) -> Node:
        return self._node(NODE_REPOSITORY, self.repository, self.repository, None)

    def _module_node(self, module: str) -> Node:
        return self._node(NODE_MODULE, module, module, self._repository_node().id)

    def _directory_node(self, directory: str, module: str) -> Node:
        if not directory or directory == module:
            return self._module_node(module)
        return self._node(
            NODE_DIRECTORY, directory, directory.split("/")[-1], self._module_node(module).id
        )

    def _add_file(self, parsed: dict) -> None:
        relative_path = parsed.get("relative_path", "")
        if not relative_path:
            return

        content = parsed.get("content", "")
        language = parsed.get("language", "unknown")

        analysis = analyse(content, relative_path, language)
        self._analyses[relative_path] = analysis

        self.stats.files += 1
        self.stats.languages[language] = self.stats.languages.get(language, 0) + 1
        if analysis.parse_failed:
            self.stats.parse_failures += 1

        module = module_of(relative_path)
        directory = directory_of(relative_path)
        parent = self._directory_node(directory, module)

        file_node = self._node(
            NODE_FILE,
            relative_path,
            parsed.get("file_name") or relative_path.split("/")[-1],
            parent.id,
            {
                "language": language,
                "extension": parsed.get("extension", ""),
                "module": module,
                "directory": directory,
                "size": len(content),
                "line_count": analysis.line_count,
                "class_count": len(analysis.classes),
                "function_count": len(analysis.functions) + len(analysis.methods),
                "import_count": len(analysis.imports),
                "complexity": analysis.complexity_band(),
                "file_hash": parsed.get("file_hash", ""),
                "parse_failed": analysis.parse_failed,
            },
        )
        self._file_ids[relative_path] = file_node.id
        self._register_module_paths(relative_path)

        for symbol in analysis.symbols:
            self._add_symbol(file_node, symbol, relative_path)

    def _add_symbol(self, file_node: Node, symbol, relative_path: str) -> None:
        kind = {
            "class": NODE_CLASS,
            "function": NODE_FUNCTION,
            "method": NODE_METHOD,
        }.get(symbol.kind, NODE_FUNCTION)

        parent_id = file_node.id
        if symbol.kind == "method" and symbol.parent:
            owner = self._nodes.get(
                node_id(self.repository, NODE_CLASS, f"{relative_path}::{symbol.parent}")
            )
            if owner is not None:
                parent_id = owner.id

        node = self._node(
            kind,
            f"{relative_path}::{symbol.qualified_name}",
            symbol.qualified_name,
            parent_id,
            {
                "relative_path": relative_path,
                "start_line": symbol.start_line,
                "end_line": symbol.end_line,
                "line_count": symbol.line_count,
                "is_async": symbol.is_async,
                "bases": symbol.bases,
            },
        )
        self.stats.symbols += 1

        # EXTENDS edges are resolved within the file's own classes; a full
        # cross-file class hierarchy needs name resolution we do not attempt.
        for base in symbol.bases:
            base_id = node_id(self.repository, NODE_CLASS, f"{relative_path}::{base}")
            if base_id in self._nodes:
                self.graph.add_edge(
                    Edge(self.repository, EDGE_EXTENDS, node.id, base_id)
                )

    # -- dependencies ------------------------------------------------------

    def _register_module_paths(self, relative_path: str) -> None:
        """Index a file under every dotted path that could import it.

        ``app/rag/parser.py`` is importable as ``app.rag.parser`` and, from
        within the package, as ``rag.parser`` or ``parser`` — repositories are
        rooted differently depending on where they are installed from, so all
        suffixes are registered and the longest match wins at lookup time.
        """
        if not relative_path.endswith(".py"):
            stem = relative_path.rsplit(".", 1)[0]
            self._module_index.setdefault(stem, relative_path)
            self._module_index.setdefault(stem.split("/")[-1], relative_path)
            return

        stem = relative_path[:-3]
        if stem.endswith("/__init__"):
            stem = stem[: -len("/__init__")]
        parts = [p for p in stem.split("/") if p]
        for start in range(len(parts)):
            dotted = ".".join(parts[start:])
            if dotted:
                self._module_index.setdefault(dotted, relative_path)

    def _resolve_python_import(self, importer: str, module: str, level: int) -> str | None:
        """Map an import statement onto a file in this repository."""
        if level:
            # Relative import: walk up from the importing file's package.
            package = directory_of(importer).split("/")
            package = package[: len(package) - (level - 1)] if level > 1 else package
            candidate = "/".join([*package, *module.split(".")]) if module else "/".join(package)
            for suffix in (".py", "/__init__.py"):
                if candidate + suffix in self._file_ids:
                    return candidate + suffix
            return self._module_index.get(module) if module else None

        if module in self._module_index:
            return self._module_index[module]

        # Fall back to progressively shorter prefixes: `a.b.c.d` may be a
        # symbol `d` inside module `a.b.c`.
        parts = module.split(".")
        for end in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:end])
            if prefix in self._module_index:
                return self._module_index[prefix]
        return None

    def _resolve_relative_file(self, importer: str, target: str) -> str | None:
        """Resolve a JS/TS-style relative import such as ``./auth.service``."""
        base = directory_of(importer)
        cleaned = target
        while cleaned.startswith("./") or cleaned.startswith("../"):
            if cleaned.startswith("../"):
                base = directory_of(base)
                cleaned = cleaned[3:]
            else:
                cleaned = cleaned[2:]
        candidate = f"{base}/{cleaned}" if base else cleaned

        for suffix in ("", ".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.js"):
            if candidate + suffix in self._file_ids:
                return candidate + suffix
        return None

    def _add_dependencies(self) -> None:
        """Second pass: turn imports into IMPORTS and DEPENDS_ON edges."""
        module_pairs: set[tuple[str, str]] = set()
        externals: dict[str, int] = defaultdict(int)

        for relative_path, analysis in self._analyses.items():
            source_id = self._file_ids.get(relative_path)
            if not source_id:
                continue

            for imported in analysis.imports:
                target_path = None
                module = imported.module

                if analysis.language == "python":
                    target_path = self._resolve_python_import(
                        relative_path, module, imported.relative_level
                    )
                elif module.startswith(_RELATIVE_MARKERS):
                    target_path = self._resolve_relative_file(relative_path, module)
                else:
                    target_path = self._module_index.get(module)

                if target_path and target_path != relative_path:
                    target_id = self._file_ids[target_path]
                    self.stats.internal_imports += 1
                    self.graph.add_edge(
                        Edge(
                            self.repository,
                            EDGE_IMPORTS,
                            source_id,
                            target_id,
                            {"names": imported.names[:8], "line": imported.line},
                        )
                    )
                    source_module = module_of(relative_path)
                    target_module = module_of(target_path)
                    if source_module != target_module:
                        module_pairs.add((source_module, target_module))
                elif module and not module.startswith(_RELATIVE_MARKERS):
                    # Top-level package name is what identifies the dependency.
                    package = module.split(".")[0].split("/")[0]
                    if package:
                        externals[package] += 1
                        self.stats.external_imports += 1
                else:
                    self.stats.unresolved_imports += 1

        # Module-level DEPENDS_ON, so the top graph level has real edges
        # rather than only CONTAINS.
        for source_module, target_module in module_pairs:
            self.graph.add_edge(
                Edge(
                    self.repository,
                    EDGE_DEPENDS_ON,
                    self._module_node(source_module).id,
                    self._module_node(target_module).id,
                )
            )

        # External packages become nodes only if used more than once, which
        # keeps one-off imports from cluttering the module view.
        repository_node = self._repository_node()
        for package, uses in sorted(externals.items(), key=lambda kv: -kv[1]):
            if uses < 2:
                continue
            self._node(
                NODE_EXTERNAL, f"external/{package}", package, repository_node.id, {"uses": uses}
            )


def build_repository_graph(repository: str, parsed_files: Sequence[dict]) -> tuple[RepositoryGraph, BuildStats]:
    """Convenience wrapper used by the indexing service."""
    builder = GraphBuilder(repository)
    graph = builder.build(parsed_files)
    return graph, builder.stats
