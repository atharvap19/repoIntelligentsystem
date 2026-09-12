"""Static analysis of a source file: symbols and dependencies.

Python is analysed with ``ast`` — the same parse the chunker already performs,
so the cost is one extra tree walk, not a new toolchain. Other languages use
conservative regular expressions: enough to place a file in the graph and draw
its imports, without pretending to be a real parser.

For Python the analysis also records what the builder needs to resolve calls
without guessing: each call as the dotted chain it was written as, the names a
function binds locally (which shadow module names), and types that are stated
in the source — ``x = Foo()``, ``x: Foo``, ``self.x = Foo()``.

Adding a language means registering an analyser in ``ANALYSERS``. That is the
same extension point the chunker uses for ``strategy_for(language)``.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

#: Chain step for calling the value so far: ``Foo().bar()`` is ``("Foo", CALL, "bar")``.
CALL = "()"
#: Chain root for zero-argument ``super()``.
SUPER = "super()"
#: ``ImportInfo.scope`` for an import inside a class body.
CLASS_SCOPE = "<class>"

#: ``x = None`` — compatible with any stated type, since calling a method on
#: the variable only succeeds when it holds the other value.
_NONE = ("<None>",)

#: Annotation wrappers whose argument is the type actually held.
_TYPE_WRAPPERS = frozenset({"Optional", "ClassVar", "Final"})


@dataclass(frozen=True)
class CallSite:
    """One call expression, as written."""

    #: ``self.repo.load()`` -> ``("self", "repo", "load")``.
    chain: tuple[str, ...]
    line: int

    @property
    def name(self) -> str:
        return ".".join(self.chain).replace("." + CALL, CALL)


@dataclass
class SymbolInfo:
    """A class, function or method defined in a file."""

    name: str
    kind: str  # "class" | "function" | "method"
    start_line: int
    end_line: int
    #: Enclosing class path for a method or nested class — ``Outer.Inner`` — otherwise empty.
    parent: str = ""
    #: Base classes as written, for INHERITS edges.
    bases: list[str] = field(default_factory=list)
    is_async: bool = False
    #: Deduplicated, readable names of what this symbol calls.
    calls: list[str] = field(default_factory=list)
    call_sites: list[CallSite] = field(default_factory=list)
    #: Names bound inside the body; they shadow module-level names.
    local_names: set[str] = field(default_factory=set)
    #: Local name -> dotted type, kept only when every binding agrees.
    local_types: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: A method's instance parameter (``cls`` for a classmethod); empty otherwise.
    self_name: str = ""
    is_classmethod: bool = False
    #: Classes only: attribute -> dotted type, kept only when every binding agrees.
    attribute_types: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        return f"{self.parent}.{self.name}" if self.parent else self.name

    @property
    def line_count(self) -> int:
        return max(1, self.end_line - self.start_line + 1)


@dataclass
class ImportInfo:
    """One import statement."""

    #: Dotted module path as written: ``app.rag.parser``.
    module: str
    #: Names pulled from it: ``["RepositoryParser"]``. Empty for plain imports.
    names: list[str] = field(default_factory=list)
    #: ``from . import x`` -> 1, ``from .. import x`` -> 2, absolute -> 0.
    relative_level: int = 0
    line: int = 0
    #: Imported name (the module, for ``import a.b as c``) -> its local alias.
    aliases: dict[str, str] = field(default_factory=dict)
    #: "" at module level, the enclosing function's qualified name inside
    #: one, ``CLASS_SCOPE`` inside a class body.
    scope: str = ""

    def bound_names(self) -> list[str]:
        """Local names this statement binds."""
        if not self.names:
            return [self.aliases.get(self.module) or self.module.split(".")[0]]
        return [self.aliases.get(name, name) for name in self.names if name != "*"]


@dataclass
class FileAnalysis:
    """Everything static analysis learned about one file."""

    relative_path: str
    language: str
    line_count: int = 0
    symbols: list[SymbolInfo] = field(default_factory=list)
    imports: list[ImportInfo] = field(default_factory=list)
    parse_failed: bool = False
    #: Module-level names bound by anything other than def, class or import.
    module_assigns: set[str] = field(default_factory=set)
    #: A literal ``__all__``, or None when the module has none.
    exports: list[str] | None = None
    #: ``__all__`` is computed, so what ``import *`` pulls in is unknown.
    exports_dynamic: bool = False

    @property
    def classes(self) -> list[SymbolInfo]:
        return [s for s in self.symbols if s.kind == "class"]

    @property
    def functions(self) -> list[SymbolInfo]:
        return [s for s in self.symbols if s.kind == "function"]

    @property
    def methods(self) -> list[SymbolInfo]:
        return [s for s in self.symbols if s.kind == "method"]

    def complexity_band(self) -> str:
        """A coarse, honest proxy — not cyclomatic complexity.

        Derived from size and symbol density only. Named as a band rather than
        a number so it is not mistaken for a real metric.
        """
        symbols = len(self.symbols)
        if self.line_count > 600 or symbols > 40:
            return "high"
        if self.line_count > 200 or symbols > 12:
            return "medium"
        return "low"


# --------------------------------------------------------------------------
# Python
# --------------------------------------------------------------------------


class _PythonVisitor(ast.NodeVisitor):
    """Collects top-level symbols, their methods, and every import."""

    def __init__(self):
        self.symbols: list[SymbolInfo] = []
        self.imports: list[ImportInfo] = []
        self.module_assigns: set[str] = set()
        self.exports: list[str] | None = None
        self.exports_dynamic = False
        self._class_stack: list[str] = []
        #: Per open class: attribute -> candidate types, and attributes bound untyped.
        self._attribute_evidence: list[tuple[dict[str, set], set[str]]] = []

    def _scope(self) -> str:
        return CLASS_SCOPE if self._class_stack else ""

    # -- imports --

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.extend(_plain_imports(node, self._scope()))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.imports.append(_from_import(node, self._scope()))

    # -- module-level bindings --

    def visit_Name(self, node: ast.Name) -> None:
        if not self._class_stack and isinstance(node.ctx, (ast.Store, ast.Del)):
            self.module_assigns.add(node.id)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if not self._class_stack and node.name:
            self.module_assigns.add(node.name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if not self._class_stack:
            self._record_exports(node.targets, node.value, extend=False)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not self._class_stack and node.value is not None:
            self._record_exports([node.target], node.value, extend=False)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if not self._class_stack:
            self._record_exports([node.target], node.value, extend=True)
        self.generic_visit(node)

    def _record_exports(self, targets: list[ast.AST], value: ast.AST, extend: bool) -> None:
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            return
        names = _string_list(value)
        if names is None:
            self.exports_dynamic = True
        elif extend and self.exports is not None:
            self.exports.extend(names)
        else:
            self.exports = names

    # -- definitions --

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        symbol = SymbolInfo(
            name=node.name,
            kind="class",
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
            parent=".".join(self._class_stack),
            bases=[_name_of(base) for base in node.bases if _name_of(base)],
        )
        self.symbols.append(symbol)

        evidence = _class_body_types(node)
        self._class_stack.append(node.name)
        self._attribute_evidence.append(evidence)
        self.generic_visit(node)
        self._attribute_evidence.pop()
        self._class_stack.pop()
        symbol.attribute_types = _agreed(*evidence)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node, is_async=True)

    def _function(self, node, is_async: bool) -> None:
        inside_class = bool(self._class_stack)
        decorators = {_name_of(d) for d in node.decorator_list}
        is_static = inside_class and "staticmethod" in decorators
        scan = _scan_function(node, bind_self=inside_class and not is_static)

        symbol = SymbolInfo(
            name=node.name,
            kind="method" if inside_class else "function",
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
            parent=".".join(self._class_stack),
            is_async=is_async,
            calls=_summarise(scan.call_sites),
            call_sites=scan.call_sites,
            local_names=scan.local_names,
            local_types=scan.local_types,
            self_name=scan.self_name,
            is_classmethod=inside_class and "classmethod" in decorators,
        )
        self.symbols.append(symbol)

        for imported in scan.imports:
            imported.scope = symbol.qualified_name
            self.imports.append(imported)
        if inside_class:
            candidates, untyped = self._attribute_evidence[-1]
            for attribute, types in scan.attribute_candidates.items():
                candidates.setdefault(attribute, set()).update(types)
            untyped |= scan.attribute_untyped
        self.module_assigns |= scan.global_assigns
        # Nested defs are deliberately not surfaced as symbols: they are
        # implementation detail of their parent, and surfacing them would
        # triple the node count for no navigational value. Their calls are
        # attributed to the parent by the scan.


@dataclass
class _FunctionScan:
    call_sites: list[CallSite] = field(default_factory=list)
    local_names: set[str] = field(default_factory=set)
    local_types: dict[str, tuple[str, ...]] = field(default_factory=dict)
    self_name: str = ""
    imports: list[ImportInfo] = field(default_factory=list)
    attribute_candidates: dict[str, set] = field(default_factory=dict)
    attribute_untyped: set[str] = field(default_factory=set)
    global_assigns: set[str] = field(default_factory=set)


def _scan_function(node: ast.FunctionDef | ast.AsyncFunctionDef, bind_self: bool) -> _FunctionScan:
    """Calls, local bindings and stated types in one function body.

    Only the body is walked: decorators and default values run in the
    enclosing scope, not when the function is called. Nested functions and
    lambdas are merged into this scope, which can only make more names count
    as local — the conservative direction.
    """
    scan = _FunctionScan()
    positional = [*node.args.posonlyargs, *node.args.args]
    if bind_self and positional:
        scan.self_name = positional[0].arg

    candidates: dict[str, set] = {}
    untyped: set[str] = set()
    typed_targets: set[int] = set()
    declared_global: set[str] = set()
    import_bound: set[str] = set()

    def bind(name: str, chain: tuple[str, ...] | None = None) -> None:
        scan.local_names.add(name)
        if chain is None:
            untyped.add(name)
        else:
            candidates.setdefault(name, set()).add(chain)

    def bind_arguments(arguments: ast.arguments) -> None:
        for arg in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs):
            bind(arg.arg, _annotation_chain(arg.annotation) if arg.annotation else None)
        for arg in (arguments.vararg, arguments.kwarg):
            if arg is not None:
                bind(arg.arg)

    bind_arguments(node.args)

    for statement in node.body:
        for child in ast.walk(statement):
            if isinstance(child, ast.Call):
                chain = _call_chain(child.func)
                if chain:
                    scan.call_sites.append(CallSite(chain, child.lineno))
            elif isinstance(child, ast.Assign):
                value_type = _value_type(child.value)
                for target in child.targets:
                    if isinstance(target, ast.Name) and value_type is not None:
                        typed_targets.add(id(target))
                        scan.local_names.add(target.id)
                        if value_type != _NONE:
                            candidates.setdefault(target.id, set()).add(value_type)
                    attribute = _self_attribute(target, scan.self_name)
                    if attribute:
                        if value_type is None:
                            scan.attribute_untyped.add(attribute)
                        elif value_type != _NONE:
                            scan.attribute_candidates.setdefault(attribute, set()).add(value_type)
            elif isinstance(child, ast.AnnAssign):
                chain = _annotation_chain(child.annotation)
                if isinstance(child.target, ast.Name) and chain:
                    typed_targets.add(id(child.target))
                    bind(child.target.id, chain)
                attribute = _self_attribute(child.target, scan.self_name)
                if attribute:
                    if chain:
                        scan.attribute_candidates.setdefault(attribute, set()).add(chain)
                    else:
                        scan.attribute_untyped.add(attribute)
            elif isinstance(child, ast.AugAssign):
                attribute = _self_attribute(child.target, scan.self_name)
                if attribute:
                    scan.attribute_untyped.add(attribute)
            elif isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                scan.local_names.add(child.id)
                if id(child) not in typed_targets:
                    untyped.add(child.id)
            elif isinstance(child, ast.Global):
                declared_global.update(child.names)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bind(child.name)
                bind_arguments(child.args)
            elif isinstance(child, ast.Lambda):
                bind_arguments(child.args)
            elif isinstance(child, ast.ClassDef):
                bind(child.name)
            elif isinstance(child, ast.ExceptHandler) and child.name:
                bind(child.name)
            elif isinstance(child, (ast.MatchAs, ast.MatchStar)) and child.name:
                bind(child.name)
            elif isinstance(child, ast.MatchMapping) and child.rest:
                bind(child.rest)
            elif isinstance(child, ast.Import):
                for imported in _plain_imports(child, ""):
                    scan.imports.append(imported)
                    import_bound.update(imported.bound_names())
            elif isinstance(child, ast.ImportFrom):
                imported = _from_import(child, "")
                scan.imports.append(imported)
                import_bound.update(imported.bound_names())

    for name in declared_global:
        if name in scan.local_names or name in import_bound:
            scan.global_assigns.add(name)
        scan.local_names.discard(name)

    scan.local_types = {
        name: next(iter(chains))
        for name, chains in candidates.items()
        if len(chains) == 1
        and name not in untyped
        and name not in declared_global
        and name not in import_bound
    }
    return scan


def _class_body_types(node: ast.ClassDef) -> tuple[dict[str, set], set[str]]:
    """Attribute types stated directly in a class body: ``x: Foo``, ``x = Foo()``."""
    candidates: dict[str, set] = {}
    untyped: set[str] = set()
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            chain = _annotation_chain(statement.annotation)
            if chain:
                candidates.setdefault(statement.target.id, set()).add(chain)
            else:
                untyped.add(statement.target.id)
        elif isinstance(statement, ast.Assign):
            value_type = _value_type(statement.value)
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    if value_type is None:
                        untyped.add(target.id)
                    elif value_type != _NONE:
                        candidates.setdefault(target.id, set()).add(value_type)
                else:
                    untyped.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
    return candidates, untyped


def _agreed(candidates: dict[str, set], untyped: set[str]) -> dict[str, tuple[str, ...]]:
    return {
        name: next(iter(chains))
        for name, chains in candidates.items()
        if len(chains) == 1 and name not in untyped
    }


def _plain_imports(node: ast.Import, scope: str) -> list[ImportInfo]:
    return [
        ImportInfo(
            module=alias.name,
            line=node.lineno,
            aliases={alias.name: alias.asname} if alias.asname else {},
            scope=scope,
        )
        for alias in node.names
    ]


def _from_import(node: ast.ImportFrom, scope: str) -> ImportInfo:
    return ImportInfo(
        module=node.module or "",
        names=[a.name for a in node.names],
        relative_level=node.level or 0,
        line=node.lineno,
        aliases={a.name: a.asname for a in node.names if a.asname},
        scope=scope,
    )


def _name_of(node: ast.AST) -> str:
    """Best-effort dotted name for an expression used as a base class."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name_of(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Subscript):  # Generic[T]
        return _name_of(node.value)
    return ""


def _call_chain(node: ast.AST) -> tuple[str, ...] | None:
    """The dotted chain a call target was written as, or None if it is not one.

    Unlike ``_name_of`` this refuses anything it cannot follow — a subscript,
    a literal, an arbitrary expression — instead of dropping it, so
    ``items[0].save()`` never becomes a call to some ``save``.
    """
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        base = _call_chain(node.value)
        return (*base, node.attr) if base else None
    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "super"
            and not node.args
            and not node.keywords
        ):
            return (SUPER,)
        base = _call_chain(node.func)
        return (*base, CALL) if base else None
    return None


def _value_type(value: ast.AST) -> tuple[str, ...] | None:
    """The callee of ``x = Foo(...)``; ``_NONE`` for ``x = None``; else None."""
    if isinstance(value, ast.Constant) and value.value is None:
        return _NONE
    if isinstance(value, ast.Call):
        return _call_chain(value.func)
    return None


def _annotation_chain(annotation: ast.AST | None) -> tuple[str, ...] | None:
    """The single class an annotation names: ``Foo``, ``"Foo"``, ``Optional[Foo]``, ``Foo | None``."""
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            annotation = ast.parse(annotation.value, mode="eval").body
        except (SyntaxError, ValueError):
            return None
    if isinstance(annotation, (ast.Name, ast.Attribute)):
        return _call_chain(annotation)
    if isinstance(annotation, ast.Subscript):
        base = _call_chain(annotation.value)
        if base and base[-1] in _TYPE_WRAPPERS:
            return _annotation_chain(annotation.slice)
        return None
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        if _is_none(annotation.right):
            return _annotation_chain(annotation.left)
        if _is_none(annotation.left):
            return _annotation_chain(annotation.right)
    return None


def _is_none(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _self_attribute(target: ast.AST, self_name: str) -> str:
    """``attr`` when ``target`` is ``self.attr``; otherwise empty."""
    if (
        self_name
        and isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == self_name
    ):
        return target.attr
    return ""


def _string_list(value: ast.AST) -> list[str] | None:
    if isinstance(value, (ast.List, ast.Tuple)) and all(
        isinstance(e, ast.Constant) and isinstance(e.value, str) for e in value.elts
    ):
        return [e.value for e in value.elts]
    return None


def _summarise(call_sites: list[CallSite], limit: int = 40) -> list[str]:
    """Readable call names, deduplicated and capped."""
    found: list[str] = []
    for site in call_sites:
        name = site.name
        if name not in found:
            found.append(name)
            if len(found) >= limit:
                break
    return found


def analyse_python(source: str, relative_path: str) -> FileAnalysis:
    analysis = FileAnalysis(
        relative_path=relative_path,
        language="python",
        line_count=source.count("\n") + 1,
    )
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        # Same fallback contract as the chunker: degrade, never raise.
        analysis.parse_failed = True
        return analysis

    visitor = _PythonVisitor()
    visitor.visit(tree)
    analysis.symbols = visitor.symbols
    analysis.imports = visitor.imports
    analysis.module_assigns = visitor.module_assigns
    analysis.exports = visitor.exports
    analysis.exports_dynamic = visitor.exports_dynamic
    return analysis


# --------------------------------------------------------------------------
# Regex analysers for the other supported languages
# --------------------------------------------------------------------------

_JS_IMPORT = re.compile(
    r"""(?:^|\n)\s*(?:import\s+(?:[\w*\s{},]+\s+from\s+)?|export\s+\*?\s*from\s+)?["']([^"']+)["']\s*;?"""
)
_JS_REQUIRE = re.compile(r"""require\(\s*["']([^"']+)["']\s*\)""")
_JS_CLASS = re.compile(r"(?:^|\n)\s*(?:export\s+)?(?:default\s+)?class\s+(\w+)(?:\s+extends\s+([\w.]+))?")
_JS_FUNCTION = re.compile(
    r"(?:^|\n)\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)"
    r"|(?:^|\n)\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\("
)
_JAVA_LIKE_IMPORT = re.compile(r"(?:^|\n)\s*(?:import|using|#include)\s+[\"<]?([\w.:/]+)[\">]?\s*;?")
_JAVA_LIKE_CLASS = re.compile(
    r"(?:^|\n)\s*(?:public|private|protected|internal|abstract|final|sealed|\s)*"
    r"(?:class|interface|struct|record|enum)\s+(\w+)"
)
_GO_IMPORT = re.compile(r'(?:^|\n)\s*(?:import\s+)?(?:[\w.]+\s+)?"([\w./-]+)"')
_GO_FUNC = re.compile(r"(?:^|\n)func\s+(?:\([^)]*\)\s*)?(\w+)")
_RUST_USE = re.compile(r"(?:^|\n)\s*use\s+([\w:]+)")
_RUST_ITEM = re.compile(r"(?:^|\n)\s*(?:pub\s+)?(?:fn|struct|enum|trait)\s+(\w+)")


def _line_of(source: str, position: int) -> int:
    return source.count("\n", 0, position) + 1


def _regex_analysis(
    source: str,
    relative_path: str,
    language: str,
    import_patterns: list[re.Pattern],
    class_pattern: re.Pattern | None,
    function_pattern: re.Pattern | None,
) -> FileAnalysis:
    analysis = FileAnalysis(
        relative_path=relative_path,
        language=language,
        line_count=source.count("\n") + 1,
    )

    seen_imports: set[str] = set()
    for pattern in import_patterns:
        for match in pattern.finditer(source):
            module = (match.group(1) or "").strip()
            if module and module not in seen_imports:
                seen_imports.add(module)
                analysis.imports.append(
                    ImportInfo(module=module, line=_line_of(source, match.start()))
                )

    if class_pattern:
        for match in class_pattern.finditer(source):
            name = match.group(1)
            if not name:
                continue
            bases = [match.group(2)] if match.lastindex and match.lastindex >= 2 and match.group(2) else []
            analysis.symbols.append(
                SymbolInfo(
                    name=name,
                    kind="class",
                    start_line=_line_of(source, match.start()),
                    end_line=_line_of(source, match.start()),
                    bases=[b for b in bases if b],
                )
            )

    if function_pattern:
        for match in function_pattern.finditer(source):
            name = next((g for g in match.groups() if g), None)
            if not name:
                continue
            analysis.symbols.append(
                SymbolInfo(
                    name=name,
                    kind="function",
                    start_line=_line_of(source, match.start()),
                    end_line=_line_of(source, match.start()),
                )
            )

    return analysis


def _js(source: str, path: str, language: str) -> FileAnalysis:
    return _regex_analysis(
        source, path, language, [_JS_IMPORT, _JS_REQUIRE], _JS_CLASS, _JS_FUNCTION
    )


ANALYSERS: dict[str, Callable[[str, str, str], FileAnalysis]] = {
    "python": lambda s, p, _l: analyse_python(s, p),
    "javascript": _js,
    "typescript": _js,
    "java": lambda s, p, l: _regex_analysis(s, p, l, [_JAVA_LIKE_IMPORT], _JAVA_LIKE_CLASS, None),
    "csharp": lambda s, p, l: _regex_analysis(s, p, l, [_JAVA_LIKE_IMPORT], _JAVA_LIKE_CLASS, None),
    "kotlin": lambda s, p, l: _regex_analysis(s, p, l, [_JAVA_LIKE_IMPORT], _JAVA_LIKE_CLASS, None),
    "cpp": lambda s, p, l: _regex_analysis(s, p, l, [_JAVA_LIKE_IMPORT], _JAVA_LIKE_CLASS, None),
    "c": lambda s, p, l: _regex_analysis(s, p, l, [_JAVA_LIKE_IMPORT], None, None),
    "go": lambda s, p, l: _regex_analysis(s, p, l, [_GO_IMPORT], None, _GO_FUNC),
    "rust": lambda s, p, l: _regex_analysis(s, p, l, [_RUST_USE], None, _RUST_ITEM),
}


def analyse(source: str, relative_path: str, language: str) -> FileAnalysis:
    """Analyse one file. Never raises — unknown languages yield an empty result."""
    analyser = ANALYSERS.get(language)
    if analyser is None:
        return FileAnalysis(
            relative_path=relative_path,
            language=language,
            line_count=source.count("\n") + 1,
        )
    try:
        return analyser(source, relative_path, language)
    except Exception:
        logger.warning("Analysis failed for %s", relative_path, exc_info=True)
        return FileAnalysis(
            relative_path=relative_path,
            language=language,
            line_count=source.count("\n") + 1,
            parse_failed=True,
        )
