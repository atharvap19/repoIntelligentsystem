"""Repository chunker.

Splits parsed files into the units that get embedded.

Python files are chunked along their syntax tree, so a chunk is a function, a
method, a class or a run of module-level code rather than an arbitrary 1000
characters. Every other language — and any Python file that fails to parse —
falls back to the original character strategy.

The dispatch table at ``RepositoryChunker.strategy_for`` is the extension
point: adding Tree-sitter support later means registering a strategy per
language, not touching the pipeline.
"""

from __future__ import annotations

import ast
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

logger = logging.getLogger(__name__)


# Chunk kinds. Stored as metadata, so keep the values stable.
CHUNK_MODULE = "module"
CHUNK_CLASS = "class"
CHUNK_FUNCTION = "function"
CHUNK_METHOD = "method"
CHUNK_BLOCK = "block"  # character fallback

#: Metadata keys every chunk must carry, whatever strategy produced it.
CHUNK_FIELDS = (
    "repository",
    "language",
    "extension",
    "file_name",
    "relative_path",
    "file_hash",
    "chunk_index",
    "chunk_type",
    "symbol",
    "start_line",
    "end_line",
    "content",
)


@dataclass(frozen=True)
class ChunkerConfig:
    """Sizing rules.

    ``max_chunk_chars`` is deliberately larger than Phase 1's 1000: a logical
    unit is worth keeping whole, and nomic-embed-text has the context for it.

    It is a *soft* ceiling. Oversized units are split on line boundaries so
    ``start_line`` and ``end_line`` stay truthful, which means a chunk can
    exceed the limit by up to the length of one source line — and a single
    very long line (a data literal, a minified fragment) cannot be split at
    all. On FastAPI this leaves p99 at ~2064 against a 2000 ceiling.
    """

    max_chunk_chars: int = 2000
    #: Character overlap for the fallback strategy only.
    overlap: int = 100
    #: Overlapping lines when an oversized AST unit has to be split.
    overlap_lines: int = 2
    #: Fragments shorter than this are dropped as noise.
    min_chunk_chars: int = 24


@dataclass
class ChunkStats:
    """Counts describing a chunking run."""

    files_chunked: int = 0
    chunks_created: int = 0
    ast_parsed: int = 0
    ast_failed: int = 0
    fallback_files: int = 0
    oversized_units_split: int = 0
    seconds: float = 0.0
    by_type: Counter = field(default_factory=Counter)
    syntax_errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        kinds = ", ".join(f"{k}={v}" for k, v in self.by_type.most_common())
        return (
            f"{self.chunks_created} chunks from {self.files_chunked} files "
            f"({self.ast_parsed} via AST, {self.fallback_files} via fallback) "
            f"in {self.seconds:.2f}s [{kinds}]"
        )


# A syntactic unit located in the source, before it becomes a chunk dict.
@dataclass(frozen=True)
class _Unit:
    chunk_type: str
    symbol: str
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive


class RepositoryChunker:
    """Turns parsed files into embeddable chunks."""

    def __init__(self, config: ChunkerConfig | None = None):
        self.config = config or ChunkerConfig()
        self.last_stats = ChunkStats()

    # -- public API --------------------------------------------------------

    def chunk_repository(self, parsed_files: Sequence[dict] | Iterator[dict]) -> list[dict]:
        """Chunk every parsed file.

        Args:
            parsed_files: Output of ``RepositoryParser``.

        Returns:
            list[dict]: chunks carrying every key in ``CHUNK_FIELDS``.
        """
        stats = ChunkStats()
        self.last_stats = stats
        started = time.perf_counter()

        chunks: list[dict] = []
        for parsed_file in parsed_files:
            chunks.extend(self._chunk_file(parsed_file, stats))

        stats.seconds = time.perf_counter() - started
        stats.chunks_created = len(chunks)
        logger.info("Chunked repository: %s", stats.summary())
        return chunks

    def chunk_file(self, parsed_file: dict) -> list[dict]:
        """Chunk a single parsed file. Useful for incremental reindexing."""
        return self._chunk_file(parsed_file, self.last_stats)

    def strategy_for(self, language: str) -> Callable[[str], list[_Unit]] | None:
        """Return the unit extractor for a language, or ``None`` to fall back.

        Register additional languages here as Tree-sitter grammars land.
        """
        if language == "python":
            return self._python_units
        return None

    # -- orchestration -----------------------------------------------------

    def _chunk_file(self, parsed_file: dict, stats: ChunkStats) -> list[dict]:
        source = parsed_file.get("content", "")
        if not source.strip():
            return []

        stats.files_chunked += 1
        language = parsed_file.get("language", "unknown")
        strategy = self.strategy_for(language)

        units: list[_Unit] | None = None
        if strategy is not None:
            try:
                units = strategy(source)
                stats.ast_parsed += 1
            except SyntaxError as exc:
                # Python 2 files, templates, deliberate syntax-error fixtures.
                stats.ast_failed += 1
                stats.syntax_errors.append(
                    f"{parsed_file.get('relative_path', '?')}:{exc.lineno}: {exc.msg}"
                )
            except (ValueError, RecursionError) as exc:
                # Null bytes slipped past the parser, or absurd nesting depth.
                stats.ast_failed += 1
                stats.syntax_errors.append(f"{parsed_file.get('relative_path', '?')}: {exc}")

        if units is None:
            stats.fallback_files += 1
            return self._character_chunks(parsed_file, source, stats)

        return self._chunks_from_units(parsed_file, source, units, stats)

    def _chunks_from_units(
        self,
        parsed_file: dict,
        source: str,
        units: Sequence[_Unit],
        stats: ChunkStats,
    ) -> list[dict]:
        lines = source.splitlines(keepends=True)
        chunks: list[dict] = []

        for unit in units:
            for piece in self._fit_to_size(unit, lines, stats):
                content = "".join(lines[piece.start_line - 1 : piece.end_line]).strip("\n")
                if len(content.strip()) < self.config.min_chunk_chars:
                    continue
                chunks.append(
                    self._make_chunk(
                        parsed_file,
                        chunk_index=len(chunks),
                        chunk_type=piece.chunk_type,
                        symbol=piece.symbol,
                        start_line=piece.start_line,
                        end_line=piece.end_line,
                        content=content,
                    )
                )
                stats.by_type[piece.chunk_type] += 1

        # A file of nothing but imports can produce no unit worth keeping.
        if not chunks:
            return self._character_chunks(parsed_file, source, stats)
        return chunks

    def _fit_to_size(
        self, unit: _Unit, lines: Sequence[str], stats: ChunkStats
    ) -> list[_Unit]:
        """Split a unit that exceeds the ceiling, on line boundaries."""
        size = sum(len(line) for line in lines[unit.start_line - 1 : unit.end_line])
        if size <= self.config.max_chunk_chars:
            return [unit]

        stats.oversized_units_split += 1

        pieces: list[_Unit] = []
        current_start = unit.start_line
        current_size = 0

        for lineno in range(unit.start_line, unit.end_line + 1):
            current_size += len(lines[lineno - 1])
            if current_size >= self.config.max_chunk_chars and lineno > current_start:
                pieces.append(_Unit(unit.chunk_type, unit.symbol, current_start, lineno))
                # Overlap a couple of lines so a split never severs context.
                current_start = max(current_start + 1, lineno - self.config.overlap_lines + 1)
                current_size = sum(
                    len(lines[n - 1]) for n in range(current_start, lineno + 1)
                )

        # Emit a tail only if it covers lines no earlier piece reached. Without
        # this, a split landing exactly on the final line appends a redundant
        # overlap-sized duplicate of that line.
        covered_to = pieces[-1].end_line if pieces else unit.start_line - 1
        if unit.end_line > covered_to:
            pieces.append(_Unit(unit.chunk_type, unit.symbol, current_start, unit.end_line))

        return pieces

    # -- python strategy ---------------------------------------------------

    def _python_units(self, source: str) -> list[_Unit]:
        """Locate module, class, function and method units via ``ast``.

        Raises:
            SyntaxError: propagated so the caller can fall back.
        """
        tree = ast.parse(source)
        lines = source.splitlines(keepends=True)

        units: list[_Unit] = []
        module_run: list[ast.stmt] = []

        def flush_module_run() -> None:
            """Emit buffered top-level statements as one module chunk."""
            if not module_run:
                return
            start = min(self._span(node)[0] for node in module_run)
            end = max(self._span(node)[1] for node in module_run)
            units.append(_Unit(CHUNK_MODULE, "", start, end))
            module_run.clear()

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                flush_module_run()
                start, end = self._span(node)
                start = self._absorb_leading_comments(lines, start)
                units.append(_Unit(CHUNK_FUNCTION, node.name, start, end))
            elif isinstance(node, ast.ClassDef):
                flush_module_run()
                units.extend(self._class_units(node, lines))
            else:
                module_run.append(node)

        flush_module_run()
        units.sort(key=lambda u: u.start_line)
        return units

    @staticmethod
    def _absorb_leading_comments(lines: Sequence[str], start_line: int) -> int:
        """Pull contiguous ``#`` comments directly above a definition into it.

        A comment block explaining a function is part of that function as far
        as retrieval is concerned; without this it lands between units and is
        dropped entirely.
        """
        lineno = start_line
        while lineno > 1 and lines[lineno - 2].lstrip().startswith("#"):
            lineno -= 1
        return lineno

    def _class_units(self, node: ast.ClassDef, lines: Sequence[str]) -> list[_Unit]:
        """One unit for a small class; header plus per-method units for a big one."""
        start, end = self._span(node)
        start = self._absorb_leading_comments(lines, start)
        size = sum(len(line) for line in lines[start - 1 : end])

        if size <= self.config.max_chunk_chars:
            return [_Unit(CHUNK_CLASS, node.name, start, end)]

        methods = [
            child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if not methods:
            return [_Unit(CHUNK_CLASS, node.name, start, end)]

        units: list[_Unit] = []
        first_method_start = min(self._span(m)[0] for m in methods)

        # Signature, docstring and class-level attributes stay together and
        # keep the class name, so "what is APIRouter" still retrieves it.
        if first_method_start > start:
            units.append(_Unit(CHUNK_CLASS, node.name, start, first_method_start - 1))

        for method in methods:
            m_start, m_end = self._span(method)
            m_start = max(
                first_method_start, self._absorb_leading_comments(lines, m_start)
            )
            units.append(
                _Unit(CHUNK_METHOD, f"{node.name}.{method.name}", m_start, m_end)
            )

        # Anything declared after the final method (nested classes, attributes).
        last_method_end = max(self._span(m)[1] for m in methods)
        if end > last_method_end:
            units.append(_Unit(CHUNK_CLASS, node.name, last_method_end + 1, end))

        return units

    @staticmethod
    def _span(node: ast.AST) -> tuple[int, int]:
        """Inclusive 1-based line span of a node, decorators included.

        ``node.lineno`` points at ``def``/``class``, so a decorated function
        would otherwise lose its ``@app.get(...)`` line — which is exactly the
        text that makes a route handler findable.
        """
        start = getattr(node, "lineno", 1)
        decorators = getattr(node, "decorator_list", None)
        if decorators:
            start = min(start, min(d.lineno for d in decorators))
        end = getattr(node, "end_lineno", None) or start
        return start, end

    # -- fallback strategy -------------------------------------------------

    def _character_chunks(
        self, parsed_file: dict, source: str, stats: ChunkStats
    ) -> list[dict]:
        """Phase 1's fixed-size strategy, kept intact for non-Python files."""
        content = source.strip()
        if not content:
            return []

        size = self.config.max_chunk_chars
        stride = max(1, size - self.config.overlap)
        chunks: list[dict] = []

        for start in range(0, len(content), stride):
            piece = content[start : start + size]
            stripped = piece.strip()
            if len(stripped) < self.config.min_chunk_chars:
                continue

            # Real line numbers, so fallback chunks stay as navigable as AST ones.
            start_line = content.count("\n", 0, start) + 1
            end_line = start_line + piece.count("\n")

            chunks.append(
                self._make_chunk(
                    parsed_file,
                    chunk_index=len(chunks),
                    chunk_type=CHUNK_BLOCK,
                    symbol="",
                    start_line=start_line,
                    end_line=end_line,
                    content=stripped,
                )
            )
            stats.by_type[CHUNK_BLOCK] += 1

        return chunks

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _make_chunk(
        parsed_file: dict,
        *,
        chunk_index: int,
        chunk_type: str,
        symbol: str,
        start_line: int,
        end_line: int,
        content: str,
    ) -> dict:
        return {
            "repository": parsed_file.get("repository", ""),
            "language": parsed_file.get("language", "unknown"),
            "extension": parsed_file.get("extension", ""),
            "file_name": parsed_file.get("file_name", ""),
            "relative_path": parsed_file.get("relative_path", ""),
            # Carried through so incremental indexing can diff by file.
            "file_hash": parsed_file.get("file_hash", ""),
            "chunk_index": chunk_index,
            "chunk_type": chunk_type,
            "symbol": symbol,
            "start_line": start_line,
            "end_line": end_line,
            "content": content,
        }
