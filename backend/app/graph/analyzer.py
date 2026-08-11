"""Static analysis of a source file: symbols and dependencies.

Python is analysed with ``ast`` — the same parse the chunker already performs,
so the cost is one extra tree walk, not a new toolchain. Other languages use
conservative regular expressions: enough to place a file in the graph and draw
its imports, without pretending to be a real parser.

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


@dataclass
class SymbolInfo:
    """A class, function or method defined in a file."""

    name: str
    kind: str  # "class" | "function" | "method"
    start_line: int
    end_line: int
    #: ``ClassName`` for a method, otherwise empty.
    parent: str = ""
    #: Base classes, for EXTENDS edges.
    bases: list[str] = field(default_factory=list)
    is_async: bool = False
    #: Names this symbol calls, best-effort, used for CALLS edges.
    calls: list[str] = field(default_factory=list)

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


@dataclass
class FileAnalysis:
    """Everything static analysis learned about one file."""

    relative_path: str
    language: str
    line_count: int = 0
    symbols: list[SymbolInfo] = field(default_factory=list)
    imports: list[ImportInfo] = field(default_factory=list)
    parse_failed: bool = False

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
        self._class_stack: list[str] = []

    # -- imports --

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(ImportInfo(module=alias.name, line=node.lineno))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.imports.append(
            ImportInfo(
                module=node.module or "",
                names=[a.name for a in node.names],
                relative_level=node.level or 0,
                line=node.lineno,
            )
        )
        self.generic_visit(node)

    # -- definitions --

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.symbols.append(
            SymbolInfo(
                name=node.name,
                kind="class",
                start_line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
                parent=self._class_stack[-1] if self._class_stack else "",
                bases=[_name_of(base) for base in node.bases if _name_of(base)],
            )
        )
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node, is_async=True)

    def _function(self, node, is_async: bool) -> None:
        inside_class = bool(self._class_stack)
        self.symbols.append(
            SymbolInfo(
                name=node.name,
                kind="method" if inside_class else "function",
                start_line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
                parent=self._class_stack[-1] if inside_class else "",
                is_async=is_async,
                calls=_calls_within(node),
            )
        )
        # Nested defs are deliberately not descended into: they are
        # implementation detail of their parent, and surfacing them would
        # triple the node count for no navigational value.


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


def _calls_within(node: ast.AST, limit: int = 40) -> list[str]:
    """Names invoked inside a definition, deduplicated and capped."""
    found: list[str] = []
    seen: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        name = _name_of(child.func)
        if name and name not in seen:
            seen.add(name)
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
