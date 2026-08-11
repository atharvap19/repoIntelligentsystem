"""AST chunking tests (Phase 2, item 2)."""

from __future__ import annotations

import pytest

from app.rag.chunker import (
    CHUNK_BLOCK,
    CHUNK_CLASS,
    CHUNK_FIELDS,
    CHUNK_FUNCTION,
    CHUNK_METHOD,
    CHUNK_MODULE,
    ChunkerConfig,
    RepositoryChunker,
)

SAMPLE = '''\
"""Module docstring."""

import os
from pathlib import Path

CONSTANT = 42


def standalone(value):
    """A top-level function."""
    return value * 2


class Service:
    """A service."""

    registry = {}

    def __init__(self, name):
        self.name = name

    async def fetch(self, key):
        return self.registry.get(key)


@decorated
def wrapped():
    return 1
'''

# Valid-looking Python with one unparseable signature, sized like a real file
# so it clears min_chunk_chars.
BROKEN = '''\
import os
from pathlib import Path


def broken(:
    """This signature does not parse."""
    return os.getcwd()


class AlsoUnreachable:
    attribute = Path(".")
'''


def make_file(content: str, **overrides) -> dict:
    parsed = {
        "repository": "demo",
        "language": "python",
        "extension": ".py",
        "file_name": "sample.py",
        "relative_path": "pkg/sample.py",
        "content": content,
    }
    parsed.update(overrides)
    return parsed


def chunk(content: str, config: ChunkerConfig | None = None, **overrides) -> list[dict]:
    return RepositoryChunker(config).chunk_repository([make_file(content, **overrides)])


def by_symbol(chunks: list[dict]) -> dict[str, dict]:
    return {c["symbol"]: c for c in chunks if c["symbol"]}


# ------------------------------------------------------------------ metadata


def test_every_chunk_has_the_required_fields():
    for c in chunk(SAMPLE):
        assert set(c) == set(CHUNK_FIELDS)


def test_metadata_is_carried_from_the_parsed_file():
    c = chunk(SAMPLE)[0]
    assert c["repository"] == "demo"
    assert c["language"] == "python"
    assert c["extension"] == ".py"
    assert c["file_name"] == "sample.py"
    assert c["relative_path"] == "pkg/sample.py"


def test_chunk_index_is_sequential_within_a_file():
    chunks = chunk(SAMPLE)
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_metadata_values_are_chroma_compatible():
    # Chroma rejects None: every value must be str, int, float or bool.
    for c in chunk(SAMPLE):
        for key, value in c.items():
            assert isinstance(value, (str, int, float, bool)), f"{key}={value!r}"


# ------------------------------------------------------------------- units


def test_splits_into_logical_units():
    symbols = by_symbol(chunk(SAMPLE))
    assert "standalone" in symbols
    assert "wrapped" in symbols
    assert "Service" in symbols


def test_chunk_types_are_assigned():
    chunks = chunk(SAMPLE)
    types = {c["chunk_type"] for c in chunks}
    assert CHUNK_FUNCTION in types
    assert CHUNK_MODULE in types
    assert CHUNK_CLASS in types


def test_module_level_code_is_grouped_not_split_per_statement():
    module_chunks = [c for c in chunk(SAMPLE) if c["chunk_type"] == CHUNK_MODULE]
    assert len(module_chunks) == 1
    body = module_chunks[0]["content"]
    assert "import os" in body
    assert "CONSTANT = 42" in body


def test_function_body_is_not_split_when_it_fits():
    fn = by_symbol(chunk(SAMPLE))["standalone"]
    assert fn["content"].startswith("def standalone(value):")
    assert "return value * 2" in fn["content"]


def test_decorators_are_included_in_the_chunk():
    fn = by_symbol(chunk(SAMPLE))["wrapped"]
    assert fn["content"].startswith("@decorated")


def test_small_class_stays_whole():
    service = by_symbol(chunk(SAMPLE))["Service"]
    assert service["chunk_type"] == CHUNK_CLASS
    assert "def __init__" in service["content"]
    assert "async def fetch" in service["content"]


def test_large_class_splits_into_methods():
    body = "\n".join(
        f"    def method_{i}(self):\n        return {i} * {'x' * 60}\n" for i in range(30)
    )
    source = f"class Big:\n    '''Doc.'''\n\n{body}\n"
    chunks = chunk(source)
    symbols = by_symbol(chunks)

    assert "Big.method_0" in symbols
    assert "Big.method_29" in symbols
    assert symbols["Big.method_0"]["chunk_type"] == CHUNK_METHOD
    # The signature and docstring survive as their own class chunk.
    assert symbols["Big"]["chunk_type"] == CHUNK_CLASS
    assert "'''Doc.'''" in symbols["Big"]["content"]


def test_method_symbols_are_qualified():
    source = "class A:\n" + "".join(
        f"    def m{i}(self):\n        return {'y' * 80}\n" for i in range(20)
    )
    symbols = set(by_symbol(chunk(source)))
    assert "A.m0" in symbols
    assert "m0" not in symbols


def test_async_methods_are_captured():
    source = "class A:\n" + "".join(
        f"    async def m{i}(self):\n        return {'y' * 80}\n" for i in range(20)
    )
    assert "A.m3" in by_symbol(chunk(source))


def test_comments_above_a_definition_are_kept():
    source = "# explains the helper\n# second line\ndef helper():\n    return 1\n"
    fn = by_symbol(chunk(source))["helper"]
    assert "# explains the helper" in fn["content"]
    assert "# second line" in fn["content"]


# --------------------------------------------------------------- line numbers


def test_line_numbers_locate_the_symbol_in_the_source():
    source_lines = SAMPLE.splitlines()
    for c in chunk(SAMPLE):
        assert 1 <= c["start_line"] <= c["end_line"] <= len(source_lines)


def test_line_numbers_round_trip_to_the_original_text():
    lines = SAMPLE.splitlines()
    fn = by_symbol(chunk(SAMPLE))["standalone"]
    extracted = "\n".join(lines[fn["start_line"] - 1 : fn["end_line"]]).strip("\n")
    assert extracted == fn["content"]


def test_decorated_function_start_line_points_at_the_decorator():
    lines = SAMPLE.splitlines()
    fn = by_symbol(chunk(SAMPLE))["wrapped"]
    assert lines[fn["start_line"] - 1].startswith("@decorated")


# ------------------------------------------------------------------ fallback


def test_syntax_error_falls_back_to_character_chunks():
    chunker = RepositoryChunker()
    chunks = chunker.chunk_repository([make_file(BROKEN)])

    assert chunks
    assert all(c["chunk_type"] == CHUNK_BLOCK for c in chunks)
    assert chunker.last_stats.ast_failed == 1
    assert chunker.last_stats.fallback_files == 1


def test_syntax_error_is_recorded_for_diagnosis():
    chunker = RepositoryChunker()
    chunker.chunk_repository([make_file(BROKEN)])
    assert "pkg/sample.py" in chunker.last_stats.syntax_errors[0]


def test_file_below_the_minimum_produces_no_chunks():
    # Deliberate: a handful of characters is not worth an embedding.
    assert chunk("x = 1\n") == []


@pytest.mark.parametrize(
    "language,extension",
    [("javascript", ".js"), ("css", ".css"), ("sql", ".sql"), ("html", ".html")],
)
def test_non_python_uses_the_character_strategy(language: str, extension: str):
    chunks = chunk(
        "const a = 1;\n" * 400, language=language, extension=extension
    )
    assert chunks
    assert all(c["chunk_type"] == CHUNK_BLOCK for c in chunks)
    assert all(c["symbol"] == "" for c in chunks)


def test_fallback_chunks_still_carry_line_numbers():
    for c in chunk("const a = 1;\n" * 400, language="javascript", extension=".js"):
        assert c["start_line"] >= 1
        assert c["end_line"] >= c["start_line"]


def test_empty_file_produces_no_chunks():
    assert chunk("   \n\n  ") == []


# --------------------------------------------------------------------- sizing


def test_oversized_function_is_split_on_line_boundaries():
    source = "def huge():\n" + "".join(f"    x{i} = {i}\n" for i in range(400))
    config = ChunkerConfig(max_chunk_chars=500)
    chunker = RepositoryChunker(config)
    chunks = chunker.chunk_repository([make_file(source)])

    pieces = [c for c in chunks if c["symbol"] == "huge"]
    assert len(pieces) > 1
    assert all(c["chunk_type"] == CHUNK_FUNCTION for c in pieces)
    assert chunker.last_stats.oversized_units_split >= 1
    # Pieces advance through the file rather than repeating.
    starts = [c["start_line"] for c in pieces]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


def test_split_pieces_cover_the_whole_unit():
    source = "def huge():\n" + "".join(f"    x{i} = {i}\n" for i in range(400))
    pieces = [
        c
        for c in chunk(source, ChunkerConfig(max_chunk_chars=500))
        if c["symbol"] == "huge"
    ]
    assert pieces[0]["start_line"] == 1
    assert pieces[-1]["end_line"] == 401


def test_ceiling_holds_within_one_line_when_lines_are_normal():
    source = "def huge():\n" + "".join(f"    x{i} = {i}\n" for i in range(400))
    longest_line = max(len(line) for line in source.splitlines())
    for c in chunk(source, ChunkerConfig(max_chunk_chars=500)):
        assert len(c["content"]) <= 500 + longest_line


def test_ceiling_is_soft_for_an_unsplittable_long_line():
    # Splitting happens on line boundaries to keep start/end_line truthful, so
    # a single enormous line necessarily overflows the ceiling.
    source = "def f():\n    x = '" + "y" * 500 + "'\n"
    chunks = chunk(source, ChunkerConfig(max_chunk_chars=200))
    assert any(len(c["content"]) > 200 for c in chunks)


def test_chunk_size_is_configurable():
    small = chunk(SAMPLE, ChunkerConfig(max_chunk_chars=200))
    large = chunk(SAMPLE, ChunkerConfig(max_chunk_chars=100_000))
    assert len(small) >= len(large)


# ----------------------------------------------------------------- behaviour


def test_stats_are_reported():
    chunker = RepositoryChunker()
    chunker.chunk_repository([make_file(SAMPLE)])
    stats = chunker.last_stats

    assert stats.files_chunked == 1
    assert stats.ast_parsed == 1
    assert stats.chunks_created > 0
    assert sum(stats.by_type.values()) == stats.chunks_created


def test_strategy_registry_is_extensible():
    chunker = RepositoryChunker()
    assert chunker.strategy_for("python") is not None
    assert chunker.strategy_for("go") is None


def test_chunking_is_deterministic():
    assert chunk(SAMPLE) == chunk(SAMPLE)
