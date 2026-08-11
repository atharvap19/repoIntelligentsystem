"""Parser and source-filtering tests (Phase 2, item 1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.parser import (
    DEFAULT_SUPPORTED_EXTENSIONS,
    SKIP_BINARY_CONTENT,
    SKIP_EMPTY,
    SKIP_GENERATED,
    SKIP_IGNORED_EXTENSION,
    SKIP_IGNORED_FILENAME,
    SKIP_IGNORED_STEM,
    SKIP_TOO_LARGE,
    SKIP_UNSUPPORTED_EXTENSION,
    ParserConfig,
    RepositoryParser,
    content_hash,
)

from .conftest import write


def paths(files: list[dict]) -> set[str]:
    return {f["relative_path"] for f in files}


# ---------------------------------------------------------------- inclusion


def test_keeps_supported_source_files(sample_repo: Path):
    files = RepositoryParser().parse_repository(sample_repo)

    assert paths(files) == {
        "app/main.py",
        "app/rag/parser.py",
        "web/index.html",
        "web/style.css",
        "web/app.ts",
        "web/view.tsx",
        "db/schema.sql",
    }


def test_supported_extension_set_matches_spec():
    assert DEFAULT_SUPPORTED_EXTENSIONS == {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go",
        ".rs", ".cpp", ".c", ".cs", ".kt", ".sql", ".html", ".css",
    }


def test_markdown_is_excluded_by_default(sample_repo: Path):
    files = RepositoryParser().parse_repository(sample_repo)
    assert not [f for f in files if f["extension"] == ".md"]


def test_markdown_can_be_opted_back_in(sample_repo: Path):
    config = ParserConfig(supported_extensions=DEFAULT_SUPPORTED_EXTENSIONS | {".md"})
    files = RepositoryParser(config).parse_repository(sample_repo)

    # docs/guide.md returns; README/CHANGELOG stay out on their stems.
    assert "docs/guide.md" in paths(files)
    assert "README.md" not in paths(files)
    assert "CHANGELOG.md" not in paths(files)


# ---------------------------------------------------------------- exclusion


@pytest.mark.parametrize(
    "excluded",
    [
        ".git/objects/abc",
        "node_modules/left-pad/index.js",
        "__pycache__/main.cpython-311.pyc",
        "build/output.js",
        ".venv/lib/site.py",
    ],
)
def test_ignored_directories_are_pruned(sample_repo: Path, excluded: str):
    files = RepositoryParser().parse_repository(sample_repo)
    assert excluded not in paths(files)


def test_pruned_directories_are_never_walked(sample_repo: Path):
    parser = RepositoryParser()
    parser.parse_repository(sample_repo)
    stats = parser.last_stats

    # Files inside pruned trees must not even be counted as scanned.
    assert stats.directories_pruned == 5
    assert set(stats.pruned_directories) == {
        ".git", "node_modules", "__pycache__", "build", ".venv",
    }
    assert "left-pad" not in str(stats.examples)


@pytest.mark.parametrize(
    "filename,reason",
    [
        ("README.md", SKIP_IGNORED_STEM),
        ("LICENSE", SKIP_IGNORED_STEM),
        ("CHANGELOG.md", SKIP_IGNORED_STEM),
        ("mkdocs.yml", SKIP_IGNORED_FILENAME),
        (".pre-commit-config.yaml", SKIP_IGNORED_FILENAME),
        ("package-lock.json", SKIP_IGNORED_FILENAME),
    ],
)
def test_metadata_files_excluded_with_correct_reason(
    sample_repo: Path, filename: str, reason: str
):
    parser = RepositoryParser()
    parser.parse_repository(sample_repo)

    assert filename not in paths(parser.parse_repository(sample_repo))
    assert parser.last_stats.skipped[reason] >= 1


@pytest.mark.parametrize("name", ["logo.png", "icon.svg"])
def test_binary_extensions_excluded(sample_repo: Path, name: str):
    parser = RepositoryParser()
    files = parser.parse_repository(sample_repo)

    assert not [f for f in files if f["file_name"] == name]
    assert parser.last_stats.skipped[SKIP_IGNORED_EXTENSION] >= 2


def test_generated_files_excluded(sample_repo: Path):
    parser = RepositoryParser()
    files = parser.parse_repository(sample_repo)

    assert "web/vendor.min.js" not in paths(files)
    assert "proto/service_pb2.py" not in paths(files)
    assert parser.last_stats.skipped[SKIP_GENERATED] == 2


def test_unsupported_extensions_counted_separately(sample_repo: Path):
    parser = RepositoryParser()
    parser.parse_repository(sample_repo)
    # docs/guide.md is the only .md left after stem filtering removes the rest.
    assert parser.last_stats.skipped[SKIP_UNSUPPORTED_EXTENSION] >= 1


def test_empty_files_excluded(sample_repo: Path):
    parser = RepositoryParser()
    files = parser.parse_repository(sample_repo)

    assert "app/empty.py" not in paths(files)
    assert parser.last_stats.skipped[SKIP_EMPTY] == 1


def test_files_with_nul_bytes_excluded(sample_repo: Path):
    parser = RepositoryParser()
    files = parser.parse_repository(sample_repo)

    assert "app/binary.py" not in paths(files)
    assert parser.last_stats.skipped[SKIP_BINARY_CONTENT] == 1


def test_oversized_files_excluded(tmp_path: Path):
    repo = tmp_path / "big"
    write(repo, "small.py", "x = 1\n")
    write(repo, "huge.py", "# pad\n" * 200_000)

    parser = RepositoryParser(ParserConfig(max_file_bytes=1000))
    files = parser.parse_repository(repo)

    assert paths(files) == {"small.py"}
    assert parser.last_stats.skipped[SKIP_TOO_LARGE] == 1


# ---------------------------------------------------------------- metadata


def test_preserved_metadata_fields(sample_repo: Path):
    files = RepositoryParser().parse_repository(sample_repo)
    record = next(f for f in files if f["relative_path"] == "app/rag/parser.py")

    assert record == {
        "repository": "sample_repo",
        "language": "python",
        "file_name": "parser.py",
        "relative_path": "app/rag/parser.py",
        "extension": ".py",
        "file_hash": content_hash("class Parser:\n    pass\n"),
        "content": "class Parser:\n    pass\n",
    }


# ------------------------------------------------------------- content hash


def test_content_hash_is_stable():
    assert content_hash("x = 1\n") == content_hash("x = 1\n")


def test_content_hash_changes_with_content():
    assert content_hash("x = 1\n") != content_hash("x = 2\n")


def test_content_hash_ignores_line_endings():
    # The same commit checks out CRLF on Windows and LF on Linux; hashing the
    # raw text would mark every file as modified on a platform change.
    assert content_hash("a\r\nb\r\n") == content_hash("a\nb\n")
    assert content_hash("a\rb\r") == content_hash("a\nb\n")


def test_every_parsed_file_carries_a_hash(sample_repo: Path):
    for record in RepositoryParser().parse_repository(sample_repo):
        assert len(record["file_hash"]) == 32


@pytest.mark.parametrize(
    "relative,language",
    [
        ("app/main.py", "python"),
        ("web/app.ts", "typescript"),
        ("web/view.tsx", "typescript"),
        ("web/index.html", "html"),
        ("web/style.css", "css"),
        ("db/schema.sql", "sql"),
    ],
)
def test_language_detected_from_extension(sample_repo: Path, relative: str, language: str):
    files = RepositoryParser().parse_repository(sample_repo)
    record = next(f for f in files if f["relative_path"] == relative)
    assert record["language"] == language


def test_relative_paths_use_forward_slashes(sample_repo: Path):
    files = RepositoryParser().parse_repository(sample_repo)
    assert all("\\" not in f["relative_path"] for f in files)
    assert "app/rag/parser.py" in paths(files)


# ---------------------------------------------------------------- behaviour


def test_stats_account_for_every_scanned_file(sample_repo: Path):
    parser = RepositoryParser()
    parser.parse_repository(sample_repo)
    stats = parser.last_stats

    assert stats.files_scanned == stats.files_kept + stats.files_rejected
    assert stats.files_kept == 7
    assert stats.extensions[".py"] == 2
    assert stats.languages["python"] == 2
    assert stats.bytes_kept > 0


def test_iter_repository_matches_parse_repository(sample_repo: Path):
    parser = RepositoryParser()
    streamed = list(parser.iter_repository(sample_repo))
    listed = parser.parse_repository(sample_repo)
    assert paths(streamed) == paths(listed)


def test_extensions_are_configurable(sample_repo: Path):
    parser = RepositoryParser(ParserConfig(supported_extensions=frozenset({".py"})))
    files = parser.parse_repository(sample_repo)
    assert paths(files) == {"app/main.py", "app/rag/parser.py"}


def test_ignored_directories_are_configurable(sample_repo: Path):
    parser = RepositoryParser(ParserConfig(ignored_directories=frozenset({".git"})))
    files = parser.parse_repository(sample_repo)
    # node_modules is no longer pruned once it leaves the ignore set.
    assert "node_modules/left-pad/index.js" in paths(files)


def test_missing_repository_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        RepositoryParser().parse_repository(tmp_path / "nope")


def test_file_instead_of_directory_raises(tmp_path: Path):
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        RepositoryParser().parse_repository(target)
