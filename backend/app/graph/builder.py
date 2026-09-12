"""Builds a repository graph from parsed files.

Consumes ``RepositoryParser`` output directly — the same filtered file stream
the chunker sees — so the graph and the vector index always describe the same
corpus. A file excluded from retrieval is excluded from the graph.

Module assignment is the one judgement call here. A "module" is the top-level
directory a file lives under, except that repositories which put everything in
a single source root (``src/``, ``lib/``, ``app/``) get their second level
instead, because otherwise every file lands in one module and the graph says
nothing.

Relationships (IMPORTS, CALLS, INHERITS, IMPLEMENTS) are resolved statically
and conservatively: an edge is drawn only when a name can be followed through
the source to a definition in this repository. Anything that would need a
guess — an untyped receiver, a shadowed name, a lookup that passes through a
base class from outside the repository — produces no edge.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from app.graph.analyzer import CALL, CLASS_SCOPE, SUPER, FileAnalysis, ImportInfo, SymbolInfo, analyse
from app.graph.model import (
    EDGE_CALLS,
    EDGE_CONTAINS,
    EDGE_DEPENDS_ON,
    EDGE_IMPLEMENTS,
    EDGE_IMPORTS,
    EDGE_INHERITS,
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

#: External bases known to add no methods a subclass could be calling. Every
#: other external base is opaque: it might define the method being looked up.
_TRANSPARENT_BASES = frozenset(
    {"typing.Generic", "typing.Protocol", "typing_extensions.Protocol", "abc.ABC"}
)
_PROTOCOL_BASES = frozenset({"typing.Protocol", "typing_extensions.Protocol"})


@dataclass
class BuildStats:
    files: int = 0
    symbols: int = 0
    internal_imports: int = 0
    external_imports: int = 0
    unresolved_imports: int = 0
    parse_failures: int = 0
    calls_resolved: int = 0
    calls_unresolved: int = 0
    bases_resolved: int = 0
    bases_unresolved: int = 0
    seconds: float = 0.0
    languages: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.files} files, {self.symbols} symbols, "
            f"{self.internal_imports} internal / {self.external_imports} external imports, "
            f"{self.unresolved_imports} unresolved, {self.parse_failures} parse failures, "
            f"{self.calls_resolved} calls resolved / {self.calls_unresolved} not "
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


def _is_init(path: str) -> bool:
    return path == "__init__.py" or path.endswith("/__init__.py")


def _c3_merge(sequences: list[list]) -> list | None:
    """C3 linearisation merge; None when the hierarchy is inconsistent."""
    sequences = [list(s) for s in sequences if s]
    merged: list = []
    while sequences:
        for sequence in sequences:
            head = sequence[0]
            if not any(head in other[1:] for other in sequences):
                break
        else:
            return None
        merged.append(head)
        sequences = [s[1:] if s[0] == head else s for s in sequences]
        sequences = [s for s in sequences if s]
    return merged


# Resolution values are plain tuples:
#   ("module", path)  ("package", dotted, importer)  ("class" | "function" | "method", node id)
#   ("instance", class id)  ("super", class id)  ("external", dotted)
#   ("symbol", path, name) — an import not yet followed  ("assigned",) — a data binding
# and None when a name cannot be resolved with confidence.


@dataclass
class _PyModule:
    path: str
    analysis: FileAnalysis
    #: Module namespace: name -> resolution value, None when bound ambiguously.
    scope: dict[str, tuple | None] = field(default_factory=dict)
    #: Module-level import bindings awaiting ``_build_scopes``.
    pending: list[tuple[str, tuple | None]] = field(default_factory=list)
    #: Modules pulled in by ``from x import *``, in source order.
    star_sources: list[str] = field(default_factory=list)
    #: Function qualified name -> names bound by imports inside it.
    function_imports: dict[str, dict[str, tuple | None]] = field(default_factory=dict)


@dataclass
class _ClassInfo:
    id: str
    path: str
    symbol: SymbolInfo
    parent_class: str | None = None
    #: Name -> ("method" | "class", node id) for what the class body defines.
    members: dict[str, tuple] = field(default_factory=dict)
    #: Resolved bases in order: class ids, or ("opaque", name) placeholders.
    bases: list = field(default_factory=list)
    #: (base as written, class id) for each base found in the repository.
    resolved_bases: list[tuple[str, str]] = field(default_factory=list)
    is_protocol: bool = False


@dataclass
class _Scope:
    path: str
    symbol: SymbolInfo | None = None
    #: Class id of the method's class.
    owner: str | None = None
    imports: dict[str, tuple | None] = field(default_factory=dict)


class GraphBuilder:
    """Turns parsed files into nodes and edges."""

    def __init__(self, repository: str):
        self.repository = repository
        self.graph = RepositoryGraph(repository=repository)
        self.stats = BuildStats()

        self._nodes: dict[str, Node] = {}
        #: relative_path -> file node id, for import resolution.
        self._file_ids: dict[str, str] = {}
        #: Dotted module path -> relative_path, for non-Python imports.
        self._module_index: dict[str, str] = {}
        self._analyses: dict[str, FileAnalysis] = {}

        self._py: dict[str, _PyModule] = {}
        self._classes: dict[str, _ClassInfo] = {}
        self._packages: set[str] = set()
        self._package_roots: list[str] = []
        self._roots_cache: dict[str, list[str]] = {}
        self._suffix_index: dict[str, set[str]] | None = None
        self._explicit_cache: dict[str, set[str]] = {}
        self._lookups: dict[tuple[str, str], tuple | None] = {}
        self._mro_cache: dict[str, list | None] = {}

    # -- public ------------------------------------------------------------

    def build(self, parsed_files: Iterable[dict]) -> RepositoryGraph:
        """Structure first, then relationships.

        Nothing can be resolved until every file is known, so the first pass
        only creates nodes; the later passes read the whole corpus.
        """
        import time

        started = time.perf_counter()

        for parsed in parsed_files:
            self._add_file(parsed)

        self._index_python()
        self._add_dependencies()
        self._build_scopes()
        self._add_inheritance()
        self._add_calls()

        self.stats.seconds = time.perf_counter() - started
        logger.info("Built graph for %r: %s", self.repository, self.stats.summary())
        return self.graph

    # -- structure ---------------------------------------------------------

    def _node(
        self,
        kind: str,
        key: str,
        name: str,
        parent_id: str | None,
        data: dict | None = None,
        edge_data: dict | None = None,
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
                Edge(self.repository, EDGE_CONTAINS, parent_id, identifier, edge_data or {})
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
            self._add_symbol(file_node, symbol, relative_path, language)

    def _add_symbol(self, file_node: Node, symbol: SymbolInfo, relative_path: str, language: str) -> None:
        kind = {
            "class": NODE_CLASS,
            "function": NODE_FUNCTION,
            "method": NODE_METHOD,
        }.get(symbol.kind, NODE_FUNCTION)

        # Methods and nested classes hang off their class.
        parent_id = file_node.id
        if symbol.parent:
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
            edge_data={"file": relative_path, "line": symbol.start_line},
        )
        self.stats.symbols += 1

        # Regex analysers only see base names, so other languages resolve
        # inheritance within the file. Python resolves it across files later.
        if language != "python":
            for base in symbol.bases:
                base_id = node_id(self.repository, NODE_CLASS, f"{relative_path}::{base}")
                if base_id in self._nodes:
                    self.graph.add_edge(
                        Edge(
                            self.repository,
                            EDGE_INHERITS,
                            node.id,
                            base_id,
                            {"file": relative_path, "line": symbol.start_line, "base": base},
                        )
                    )

    # -- python index ------------------------------------------------------

    def _index_python(self) -> None:
        """Package layout, per-file namespaces and class member tables."""
        for path in self._analyses:
            if _is_init(path):
                self._packages.add(directory_of(path))

        roots = set()
        for package in self._packages:
            top = package
            while top and directory_of(top) in self._packages:
                top = directory_of(top)
            roots.add(directory_of(top) if top else "")
        self._package_roots = sorted(roots)

        for path, analysis in self._analyses.items():
            if analysis.language != "python":
                continue
            self._py[path] = _PyModule(path, analysis)
            for symbol in analysis.symbols:
                identifier = self._symbol_id(path, symbol)
                if symbol.kind == "class":
                    info = self._classes.setdefault(identifier, _ClassInfo(identifier, path, symbol))
                    if symbol.parent:
                        info.parent_class = self._class_id(path, symbol.parent)
                        owner = self._classes.get(info.parent_class)
                        if owner is not None:
                            owner.members.setdefault(symbol.name, ("class", identifier))
                elif symbol.kind == "method":
                    owner = self._classes.get(self._class_id(path, symbol.parent))
                    if owner is not None:
                        owner.members.setdefault(symbol.name, ("method", identifier))

    def _symbol_id(self, path: str, symbol: SymbolInfo) -> str:
        kind = {"class": NODE_CLASS, "method": NODE_METHOD}.get(symbol.kind, NODE_FUNCTION)
        return node_id(self.repository, kind, f"{path}::{symbol.qualified_name}")

    def _class_id(self, path: str, qualified_name: str) -> str:
        return node_id(self.repository, NODE_CLASS, f"{path}::{qualified_name}")

    # -- python module resolution ------------------------------------------

    def _module_file(self, root: str, dotted: str) -> str | None:
        """``root/a/b.py`` or ``root/a/b/__init__.py`` for ``a.b``."""
        base = "/".join(p for p in (root, *dotted.split("."))) if dotted else root
        base = base.strip("/")
        if dotted:
            candidates = (f"{base}.py", f"{base}/__init__.py")
        else:
            candidates = (f"{base}/__init__.py" if base else "__init__.py",)
        for candidate in candidates:
            if candidate in self._py:
                return candidate
        return None

    def _import_roots(self, importer: str) -> list[str]:
        """Directories an absolute import from ``importer`` could be rooted at.

        A file inside a package is importable from the directory above its
        topmost package; a file outside any package is run from its own
        directory. Either way the project may be launched from any ancestor,
        so those follow, nearest first, then the roots of other packages.
        """
        directory = directory_of(importer)
        cached = self._roots_cache.get(directory)
        if cached is not None:
            return cached

        anchor = directory
        while anchor and anchor in self._packages:
            anchor = directory_of(anchor)

        roots = []
        current = anchor
        while True:
            roots.append(current)
            if not current:
                break
            current = directory_of(current)
        roots += [r for r in self._package_roots if r not in roots]
        self._roots_cache[directory] = roots
        return roots

    def _suffixes(self) -> dict[str, set[str]]:
        """Dotted suffixes of two or more parts -> files, for namespace layouts."""
        if self._suffix_index is None:
            index: dict[str, set[str]] = defaultdict(set)
            for path in self._py:
                stem = path[:-3]
                if _is_init(path):
                    stem = directory_of(path)
                parts = [p for p in stem.split("/") if p]
                for start in range(len(parts) - 1):
                    index[".".join(parts[start:])].add(path)
            self._suffix_index = index
        return self._suffix_index

    def _resolve_python_module(self, importer: str, dotted: str, level: int) -> str | None:
        """The repository file an import names, or None if it is not one."""
        if level:
            parts = [p for p in directory_of(importer).split("/") if p]
            if level - 1 > len(parts):
                return None
            return self._module_file("/".join(parts[: len(parts) - (level - 1)]), dotted)

        if not dotted:
            return None
        for root in self._import_roots(importer):
            found = self._module_file(root, dotted)
            if found:
                return found
        # A single-part name like `json` is never matched this way: it would
        # turn every stdlib import into an edge to any file that shares a name.
        if "." in dotted:
            matches = self._suffixes().get(dotted, ())
            if len(matches) == 1:
                return next(iter(matches))
        return None

    def _explicit_names(self, path: str) -> set[str]:
        """Names a module binds itself, not counting ``import *``."""
        cached = self._explicit_cache.get(path)
        if cached is None:
            analysis = self._py[path].analysis
            cached = {s.name for s in analysis.symbols if not s.parent}
            cached |= analysis.module_assigns
            for imported in analysis.imports:
                if imported.scope == "":
                    cached.update(imported.bound_names())
            self._explicit_cache[path] = cached
        return cached

    def _bind_import(
        self, importer: str, imported: ImportInfo
    ) -> tuple[list[tuple[str, tuple | None]], list[tuple[str, str]], str | None]:
        """What one import statement binds, which files it reads, and its base module."""
        bindings: list[tuple[str, tuple | None]] = []
        targets: list[tuple[str, str]] = []

        if not imported.names:  # import a.b.c [as x]
            dotted = imported.module
            target = self._resolve_python_module(importer, dotted, 0)
            if target:
                targets.append((target, dotted))
            alias = imported.aliases.get(dotted)
            if alias:
                bindings.append((alias, ("module", target) if target else ("external", dotted)))
            else:
                # `import a.b.c` binds `a`; the rest is reached by attribute.
                head = dotted.split(".")[0]
                head_file = self._resolve_python_module(importer, head, 0)
                if head_file:
                    value = ("module", head_file)
                elif target:
                    value = ("package", head, importer)
                else:
                    value = ("external", head)
                bindings.append((head, value))
            return bindings, targets, None

        level = imported.relative_level
        base = imported.module
        base_file = self._resolve_python_module(importer, base, level) if (base or level) else None
        for name in imported.names:
            if name == "*":
                if base_file:
                    targets.append((base_file, "*"))
                continue
            local = imported.aliases.get(name, name)
            dotted = f"{base}.{name}" if base else name
            sub = self._resolve_python_module(importer, dotted, level)
            # A name the package binds itself wins over a same-named submodule,
            # exactly as `from pkg import x` behaves at runtime.
            explicit = base_file is not None and name in self._explicit_names(base_file)
            if sub and not explicit:
                targets.append((sub, name))
                bindings.append((local, ("module", sub)))
            elif base_file:
                targets.append((base_file, name))
                bindings.append((local, ("symbol", base_file, name)))
            elif level == 0:
                bindings.append((local, ("external", dotted)))
            else:
                bindings.append((local, None))
        return bindings, targets, base_file

    # -- dependencies ------------------------------------------------------

    def _register_module_paths(self, relative_path: str) -> None:
        """Index a file under every dotted path that could import it.

        ``app/rag/parser.py`` is importable as ``app.rag.parser`` and, from
        within the package, as ``rag.parser`` or ``parser`` — repositories are
        rooted differently depending on where they are installed from, so all
        suffixes are registered and the longest match wins at lookup time.
        Used for non-Python imports; Python resolves against real roots.
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
        """Turn imports into IMPORTS and DEPENDS_ON edges."""
        module_pairs: set[tuple[str, str]] = set()
        externals: dict[str, int] = defaultdict(int)

        for relative_path, analysis in self._analyses.items():
            source_id = self._file_ids.get(relative_path)
            if not source_id:
                continue

            if analysis.language == "python":
                self._python_imports(relative_path, analysis, source_id, module_pairs, externals)
                continue

            for imported in analysis.imports:
                module = imported.module
                if module.startswith(_RELATIVE_MARKERS):
                    target_path = self._resolve_relative_file(relative_path, module)
                else:
                    target_path = self._module_index.get(module)

                if target_path and target_path != relative_path:
                    self.stats.internal_imports += 1
                    self.graph.add_edge(
                        Edge(
                            self.repository,
                            EDGE_IMPORTS,
                            source_id,
                            self._file_ids[target_path],
                            {"names": imported.names[:8], "line": imported.line, "file": relative_path},
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

    def _python_imports(
        self,
        relative_path: str,
        analysis: FileAnalysis,
        source_id: str,
        module_pairs: set[tuple[str, str]],
        externals: dict[str, int],
    ) -> None:
        module = self._py[relative_path]
        edges: dict[str, dict] = {}

        for imported in analysis.imports:
            bindings, targets, base_file = self._bind_import(relative_path, imported)

            if imported.scope == "":
                module.pending.extend(bindings)
                if "*" in imported.names and base_file:
                    module.star_sources.append(base_file)
            elif imported.scope != CLASS_SCOPE:
                scoped = module.function_imports.setdefault(imported.scope, {})
                for local, value in bindings:
                    scoped[local] = value if scoped.get(local, value) == value else None

            targets = [(t, n) for t, n in targets if t != relative_path]
            if targets:
                self.stats.internal_imports += 1
                for target, name in targets:
                    entry = edges.setdefault(target, {"names": [], "line": imported.line})
                    if name not in entry["names"]:
                        entry["names"].append(name)
            elif imported.relative_level == 0 and imported.module:
                externals[imported.module.split(".")[0]] += 1
                self.stats.external_imports += 1
            else:
                self.stats.unresolved_imports += 1

        for target, entry in edges.items():
            self.graph.add_edge(
                Edge(
                    self.repository,
                    EDGE_IMPORTS,
                    source_id,
                    self._file_ids[target],
                    {"names": entry["names"][:8], "line": entry["line"], "file": relative_path},
                )
            )
            if module_of(relative_path) != module_of(target):
                module_pairs.add((module_of(relative_path), module_of(target)))

    def _build_scopes(self) -> None:
        """Each module's namespace; a name bound two different ways is ambiguous."""
        for path, module in self._py.items():
            entries: dict[str, set] = defaultdict(set)
            for symbol in module.analysis.symbols:
                if not symbol.parent:
                    kind = "class" if symbol.kind == "class" else "function"
                    entries[symbol.name].add((kind, self._symbol_id(path, symbol)))
            for local, value in module.pending:
                entries[local].add(value)
            for name in module.analysis.module_assigns:
                entries[name].add(("assigned",))
            module.scope = {
                name: next(iter(values)) if len(values) == 1 else None
                for name, values in entries.items()
            }

    # -- name resolution ---------------------------------------------------

    def _lookup_module(self, path: str, name: str) -> tuple | None:
        """What ``name`` means inside module ``path``, following re-exports."""
        key = (path, name)
        if key in self._lookups:
            return self._lookups[key]
        self._lookups[key] = None  # in progress: a re-export cycle resolves to nothing
        result = self._lookup_module_uncached(path, name)
        self._lookups[key] = result
        return result

    def _lookup_module_uncached(self, path: str, name: str) -> tuple | None:
        module = self._py.get(path)
        if module is None:
            return None
        if name in module.scope:
            return self._deref(module.scope[name])
        for source in reversed(module.star_sources):
            if self._star_exports(source, name):
                found = self._lookup_module(source, name)
                if found is not None:
                    return found
        if _is_init(path):
            submodule = self._module_file(directory_of(path), name)
            if submodule:
                return ("module", submodule)
        return None

    def _star_exports(self, path: str, name: str) -> bool:
        analysis = self._py[path].analysis
        if analysis.exports_dynamic:
            return False
        if analysis.exports is not None:
            return name in analysis.exports
        return not name.startswith("_")

    def _deref(self, value: tuple | None) -> tuple | None:
        if value is None or value[0] == "assigned":
            return None
        if value[0] == "symbol":
            return self._lookup_module(value[1], value[2])
        return value

    def _resolve_chain(self, scope: _Scope, chain: tuple[str, ...]) -> tuple | None:
        value = self._resolve_root(scope, chain[0])
        for part in chain[1:]:
            if value is None:
                return None
            value = self._attribute(value, part)
        return value

    def _resolve_root(self, scope: _Scope, name: str) -> tuple | None:
        symbol = scope.symbol
        if symbol is not None:
            if name == SUPER:
                return ("super", scope.owner) if scope.owner and symbol.self_name else None
            if scope.owner and name == symbol.self_name:
                return ("class" if symbol.is_classmethod else "instance", scope.owner)
        if name in scope.imports and (symbol is None or name not in symbol.local_names):
            return self._deref(scope.imports[name])
        if symbol is not None:
            if name in symbol.local_types:
                stated = self._resolve_chain(
                    _Scope(scope.path, imports=scope.imports), symbol.local_types[name]
                )
                return ("instance", stated[1]) if stated and stated[0] == "class" else None
            if name in symbol.local_names:
                return None
        if name == SUPER:
            return None
        return self._lookup_module(scope.path, name)

    def _attribute(self, value: tuple, part: str) -> tuple | None:
        kind = value[0]
        if kind == "module":
            return None if part == CALL else self._lookup_module(value[1], part)
        if kind == "package":
            if part == CALL:
                return None
            dotted = f"{value[1]}.{part}"
            found = self._resolve_python_module(value[2], dotted, 0)
            return ("module", found) if found else ("package", dotted, value[2])
        if kind == "class":
            return ("instance", value[1]) if part == CALL else self._member(value[1], part)
        if kind == "instance":
            if part == CALL:
                return None
            member = self._member(value[1], part)
            if member is not None:
                return member
            attribute_type = self._attribute_type(value[1], part)
            return ("instance", attribute_type) if attribute_type else None
        if kind == "super":
            return None if part == CALL else self._member(value[1], part, skip_self=True)
        if kind == "external":
            return None if part == CALL else ("external", f"{value[1]}.{part}")
        return None

    # -- classes -----------------------------------------------------------

    def _mro(self, class_id: str) -> list | None:
        """C3 linearisation, with opaque placeholders for external bases."""
        if class_id in self._mro_cache:
            return self._mro_cache[class_id]
        self._mro_cache[class_id] = None  # in progress: an inheritance cycle has no MRO

        info = self._classes[class_id]
        sequences = []
        for base in info.bases:
            if isinstance(base, tuple):
                sequences.append([base])
                continue
            parent = self._mro(base)
            if parent is None:
                return None
            sequences.append(parent)

        merged = _c3_merge([*sequences, list(info.bases)])
        result = None if merged is None else [class_id, *merged]
        self._mro_cache[class_id] = result
        return result

    def _member(self, class_id: str, name: str, skip_self: bool = False) -> tuple | None:
        """A method or nested class found through the MRO.

        Stops at the first opaque base: a class from outside the repository
        may define the name, and claiming a later repository class would be
        a guess.
        """
        mro = self._mro(class_id)
        if mro is None:
            return None
        for entry in mro[1:] if skip_self else mro:
            if isinstance(entry, tuple):
                return None
            member = self._classes[entry].members.get(name)
            if member is not None:
                return member
        return None

    def _attribute_type(self, class_id: str, name: str) -> str | None:
        mro = self._mro(class_id)
        if mro is None:
            return None
        for entry in mro:
            if isinstance(entry, tuple):
                return None
            info = self._classes[entry]
            chain = info.symbol.attribute_types.get(name)
            if chain:
                value = self._resolve_chain(_Scope(info.path), chain)
                return value[1] if value and value[0] == "class" else None
        return None

    def _resolve_bases(self, info: _ClassInfo) -> None:
        for raw in info.symbol.bases:
            chain = tuple(raw.split("."))
            value = None
            parent = self._classes.get(info.parent_class) if info.parent_class else None
            if parent is not None and chain[0] in parent.members:
                value = parent.members[chain[0]]
                for part in chain[1:]:
                    value = self._attribute(value, part) if value else None
            else:
                value = self._resolve_chain(_Scope(info.path), chain)

            if value and value[0] == "class" and value[1] != info.id:
                info.bases.append(value[1])
                info.resolved_bases.append((raw, value[1]))
                self.stats.bases_resolved += 1
                continue

            dotted = value[1] if value and value[0] == "external" else raw
            if dotted in _TRANSPARENT_BASES or (value is None and raw == "object"):
                info.is_protocol = info.is_protocol or dotted in _PROTOCOL_BASES
                continue
            info.bases.append(("opaque", dotted))
            self.stats.bases_unresolved += 1

    def _add_inheritance(self) -> None:
        """INHERITS, or IMPLEMENTS for an explicit subclass of a Protocol.

        PEP 544 makes naming a protocol as a base an explicit declaration that
        the class implements it. That is the only case Python states in
        source; structural conformance and ABC registration are not recorded.
        """
        for info in self._classes.values():
            self._resolve_bases(info)

        for info in self._classes.values():
            for raw, base_id in info.resolved_bases:
                base = self._classes[base_id]
                kind = EDGE_IMPLEMENTS if base.is_protocol and not info.is_protocol else EDGE_INHERITS
                self.graph.add_edge(
                    Edge(
                        self.repository,
                        kind,
                        info.id,
                        base_id,
                        {"file": info.path, "line": info.symbol.start_line, "base": raw},
                    )
                )

    # -- calls -------------------------------------------------------------

    def _call_target(self, scope: _Scope, chain: tuple[str, ...]) -> tuple[str, bool] | None:
        """(target node id, is_constructor) for a call, when it resolves."""
        value = self._resolve_chain(scope, chain)
        if value is None:
            return None
        if value[0] in ("function", "method"):
            return value[1], False
        if value[0] == "class":
            return value[1], True
        if value[0] == "instance":
            member = self._member(value[1], "__call__")
            if member is not None and member[0] == "method":
                return member[1], False
        return None

    def _add_calls(self) -> None:
        """CALLS from each function or method to the definitions it invokes."""
        found: dict[tuple[str, str], dict] = {}

        for path, module in self._py.items():
            for symbol in module.analysis.symbols:
                if symbol.kind == "class" or not symbol.call_sites:
                    continue
                source = self._symbol_id(path, symbol)
                owner = self._class_id(path, symbol.parent) if symbol.kind == "method" else None
                scope = _Scope(
                    path,
                    symbol,
                    owner if owner in self._classes else None,
                    module.function_imports.get(symbol.qualified_name, {}),
                )
                for site in symbol.call_sites:
                    resolved = self._call_target(scope, site.chain)
                    if resolved is None:
                        self.stats.calls_unresolved += 1
                        continue
                    self.stats.calls_resolved += 1
                    target, constructor = resolved
                    if target == source:
                        continue
                    entry = found.setdefault(
                        (source, target),
                        {"file": path, "lines": [], "via": _via(symbol, site.chain, constructor)},
                    )
                    entry["lines"].append(site.line)

        for (source, target), entry in found.items():
            lines = sorted(entry["lines"])
            self.graph.add_edge(
                Edge(
                    self.repository,
                    EDGE_CALLS,
                    source,
                    target,
                    {
                        "file": entry["file"],
                        "line": lines[0],
                        "lines": lines[:20],
                        "count": len(lines),
                        "via": entry["via"],
                    },
                )
            )


def _via(symbol: SymbolInfo, chain: tuple[str, ...], constructor: bool) -> str:
    """How a call was resolved, recorded on the edge for auditing."""
    if constructor:
        return "constructor"
    root = chain[0]
    if root == SUPER:
        return "super"
    if symbol.self_name and root == symbol.self_name:
        return "self"
    if root in symbol.local_types:
        return "local-instance"
    return "name" if len(chain) == 1 else "attribute"


def build_repository_graph(repository: str, parsed_files: Sequence[dict]) -> tuple[RepositoryGraph, BuildStats]:
    """Convenience wrapper used by the indexing service."""
    builder = GraphBuilder(repository)
    graph = builder.build(parsed_files)
    return graph, builder.stats
