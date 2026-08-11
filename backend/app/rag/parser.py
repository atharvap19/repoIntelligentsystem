"""Repository parser.

Walks a cloned repository and yields the source files worth indexing.

Filtering runs as a whitelist: an extension must be explicitly supported to be
indexed. Everything else — binaries, lockfiles, prose, generated bundles — is
skipped and counted, so a repository's composition is always explainable.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

#: Source extensions eligible for indexing. Markdown is deliberately absent —
#: see the note in ``ParserConfig``.
DEFAULT_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".go",
        ".rs",
        ".cpp",
        ".c",
        ".cs",
        ".kt",
        ".sql",
        ".html",
        ".css",
    }
)

DEFAULT_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".md": "markdown",
}

#: Directory names pruned during the walk — never descended into.
DEFAULT_IGNORED_DIRECTORIES: frozenset[str] = frozenset(
    {
        ".git",
        ".github",
        ".gitlab",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".vscode",
        ".idea",
        "__pycache__",
        "node_modules",
        "bower_components",
        "vendor",
        "dist",
        "build",
        "out",
        "target",
        ".venv",
        "venv",
        "env",
        ".env",
        ".next",
        ".nuxt",
        ".turbo",
        ".cache",
        "coverage",
        "htmlcov",
        "site-packages",
    }
)

#: Exact filenames to skip, compared case-insensitively.
DEFAULT_IGNORED_FILENAMES: frozenset[str] = frozenset(
    {
        "mkdocs.yml",
        "mkdocs.yaml",
        ".pre-commit-config.yaml",
        ".editorconfig",
        ".gitattributes",
        ".gitignore",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "cargo.lock",
        "gemfile.lock",
        "composer.lock",
    }
)

#: Filename stems to skip regardless of extension — README.md, LICENSE.txt,
#: CHANGELOG, and friends are prose, not implementation.
DEFAULT_IGNORED_STEMS: frozenset[str] = frozenset(
    {
        "readme",
        "license",
        "licence",
        "copying",
        "changelog",
        "changes",
        "history",
        "contributing",
        "code_of_conduct",
        "security",
        "notice",
        "authors",
        "maintainers",
        "codeowners",
    }
)

#: Binary and generated extensions. Redundant under the whitelist, but kept as
#: an explicit second gate so widening ``supported_extensions`` stays safe.
DEFAULT_IGNORED_EXTENSIONS: frozenset[str] = frozenset(
    {
        # images
        ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".bmp", ".webp", ".tiff",
        # documents / archives
        ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
        # compiled artefacts
        ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".exe", ".bin",
        ".o", ".a", ".obj", ".class", ".jar", ".war", ".wasm",
        # media
        ".mp3", ".mp4", ".avi", ".mov", ".webm", ".wav", ".ogg",
        # fonts
        ".woff", ".woff2", ".ttf", ".eot", ".otf",
        # data / misc
        ".lock", ".db", ".sqlite", ".sqlite3", ".parquet", ".npy", ".pkl",
        ".map",
    }
)

#: Filename suffixes that indicate machine-generated output.
DEFAULT_GENERATED_SUFFIXES: tuple[str, ...] = (
    ".min.js",
    ".min.css",
    ".bundle.js",
    ".bundle.css",
    ".js.map",
    ".css.map",
    "_pb2.py",
    "_pb2_grpc.py",
    ".generated.py",
    ".g.dart",
)

#: Source files above this size are almost always generated or vendored.
DEFAULT_MAX_FILE_BYTES = 1_000_000

# Skip reasons, used as stable keys in ``ParseStats.skipped``.
SKIP_IGNORED_DIRECTORY = "ignored_directory"
SKIP_IGNORED_FILENAME = "ignored_filename"
SKIP_IGNORED_STEM = "ignored_stem"
SKIP_IGNORED_EXTENSION = "ignored_extension"
SKIP_UNSUPPORTED_EXTENSION = "unsupported_extension"
SKIP_GENERATED = "generated_file"
SKIP_TOO_LARGE = "too_large"
SKIP_BINARY_CONTENT = "binary_content"
SKIP_EMPTY = "empty"
SKIP_READ_ERROR = "read_error"


def content_hash(content: str) -> str:
    """Stable digest of a file's text, used to detect changes.

    Line endings are normalised first. A repository cloned on Windows checks
    out CRLF while the same commit on Linux checks out LF — hashing the raw
    text would mark every file as modified purely because of the platform.
    """
    normalised = content.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:32]


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ParserConfig:
    """Filtering rules for a parse run.

    Markdown is not supported by default. Prose translations dominate
    doc-heavy repositories — in FastAPI, ``.md`` accounts for 64% of files —
    and they crowd implementation code out of the top-k at query time. Pass
    ``supported_extensions=DEFAULT_SUPPORTED_EXTENSIONS | {".md"}`` to opt back
    in for a documentation-focused corpus.
    """

    supported_extensions: frozenset[str] = DEFAULT_SUPPORTED_EXTENSIONS
    language_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LANGUAGE_MAP))
    ignored_directories: frozenset[str] = DEFAULT_IGNORED_DIRECTORIES
    ignored_filenames: frozenset[str] = DEFAULT_IGNORED_FILENAMES
    ignored_stems: frozenset[str] = DEFAULT_IGNORED_STEMS
    ignored_extensions: frozenset[str] = DEFAULT_IGNORED_EXTENSIONS
    generated_suffixes: tuple[str, ...] = DEFAULT_GENERATED_SUFFIXES
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    follow_symlinks: bool = False

    def language_for(self, extension: str) -> str:
        return self.language_map.get(extension.lower(), "unknown")


@dataclass
class ParseStats:
    """Counts describing what a parse run kept and discarded."""

    files_scanned: int = 0
    files_kept: int = 0
    directories_pruned: int = 0
    bytes_kept: int = 0
    seconds: float = 0.0
    skipped: Counter = field(default_factory=Counter)
    extensions: Counter = field(default_factory=Counter)
    languages: Counter = field(default_factory=Counter)
    pruned_directories: Counter = field(default_factory=Counter)
    #: A few example paths per skip reason, to make filtering auditable.
    examples: dict[str, list[str]] = field(default_factory=dict)

    _example_limit: int = 5

    def record_skip(self, reason: str, relative_path: str) -> None:
        self.skipped[reason] += 1
        bucket = self.examples.setdefault(reason, [])
        if len(bucket) < self._example_limit:
            bucket.append(relative_path)

    @property
    def files_rejected(self) -> int:
        return sum(self.skipped.values())

    def summary(self) -> str:
        kept_pct = (self.files_kept / self.files_scanned * 100) if self.files_scanned else 0.0
        return (
            f"{self.files_kept}/{self.files_scanned} files kept ({kept_pct:.1f}%), "
            f"{self.directories_pruned} directories pruned, "
            f"{self.bytes_kept / 1_000_000:.1f}MB in {self.seconds:.2f}s"
        )


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


class RepositoryParser:
    """Extracts indexable source files from a cloned repository."""

    def __init__(self, config: ParserConfig | None = None):
        self.config = config or ParserConfig()
        self.last_stats = ParseStats()

    # -- public API --------------------------------------------------------

    def parse_repository(self, repository_path: str | Path) -> list[dict]:
        """Parse every supported source file into a list.

        Returns
        -------
        list[dict]
            One entry per kept file, carrying ``repository``, ``language``,
            ``extension``, ``file_name``, ``relative_path`` and ``content``.
        """
        return list(self.iter_repository(repository_path))

    def iter_repository(self, repository_path: str | Path) -> Iterator[dict]:
        """Stream parsed files instead of materialising them all at once.

        Preferred by the indexing pipeline so a large repository's contents
        never sit in memory in full.
        """
        repository = Path(repository_path)
        if not repository.exists():
            raise FileNotFoundError(f"Repository not found: {repository_path}")
        if not repository.is_dir():
            raise NotADirectoryError(f"Not a directory: {repository_path}")

        stats = ParseStats()
        self.last_stats = stats
        started = time.perf_counter()

        repository_name = repository.name
        config = self.config

        for directory, subdirectories, filenames in os.walk(
            repository, followlinks=config.follow_symlinks
        ):
            # Prune in place so os.walk never descends into ignored trees.
            # This is what keeps .git and node_modules from being walked at all.
            kept_subdirectories = []
            for name in subdirectories:
                if name.lower() in config.ignored_directories:
                    stats.directories_pruned += 1
                    stats.pruned_directories[name] += 1
                else:
                    kept_subdirectories.append(name)
            subdirectories[:] = kept_subdirectories

            for filename in filenames:
                stats.files_scanned += 1
                file_path = Path(directory) / filename
                relative_path = self._relative_path(file_path, repository)

                parsed = self._parse_file(file_path, relative_path, repository_name, stats)
                if parsed is not None:
                    yield parsed

        stats.seconds = time.perf_counter() - started
        logger.info("Parsed %s: %s", repository_name, stats.summary())

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _relative_path(file_path: Path, repository: Path) -> str:
        """Repository-relative path with forward slashes on every platform.

        Windows would otherwise produce ``fastapi\\routing.py``, which leaks
        into metadata and vector IDs and makes them host-dependent.
        """
        return file_path.relative_to(repository).as_posix()

    def _parse_file(
        self,
        file_path: Path,
        relative_path: str,
        repository_name: str,
        stats: ParseStats,
    ) -> dict | None:
        """Apply every filter to one file; return its record or ``None``."""
        config = self.config
        filename = file_path.name
        lowered = filename.lower()

        if lowered in config.ignored_filenames:
            stats.record_skip(SKIP_IGNORED_FILENAME, relative_path)
            return None

        if any(lowered.endswith(suffix.lower()) for suffix in config.generated_suffixes):
            stats.record_skip(SKIP_GENERATED, relative_path)
            return None

        extension = file_path.suffix.lower()

        # Stem check runs before the whitelist so README.md is reported as
        # prose rather than as an unsupported extension.
        if self._stem_of(lowered, extension) in config.ignored_stems:
            stats.record_skip(SKIP_IGNORED_STEM, relative_path)
            return None

        if extension in config.ignored_extensions:
            stats.record_skip(SKIP_IGNORED_EXTENSION, relative_path)
            return None

        if extension not in config.supported_extensions:
            stats.record_skip(SKIP_UNSUPPORTED_EXTENSION, relative_path)
            return None

        try:
            size = file_path.stat().st_size
        except OSError:
            stats.record_skip(SKIP_READ_ERROR, relative_path)
            return None

        if size > config.max_file_bytes:
            stats.record_skip(SKIP_TOO_LARGE, relative_path)
            return None

        try:
            raw = file_path.read_bytes()
        except OSError:
            stats.record_skip(SKIP_READ_ERROR, relative_path)
            return None

        # A NUL byte means this is not text, whatever the extension claims.
        if b"\x00" in raw:
            stats.record_skip(SKIP_BINARY_CONTENT, relative_path)
            return None

        content = raw.decode("utf-8", errors="ignore")
        if not content.strip():
            stats.record_skip(SKIP_EMPTY, relative_path)
            return None

        language = config.language_for(extension)

        stats.files_kept += 1
        stats.bytes_kept += size
        stats.extensions[extension] += 1
        stats.languages[language] += 1

        return {
            "repository": repository_name,
            "language": language,
            "file_name": filename,
            "relative_path": relative_path,
            "extension": extension,
            "file_hash": content_hash(content),
            "content": content,
        }

    @staticmethod
    def _stem_of(lowered_filename: str, extension: str) -> str:
        """``readme`` from ``README.md``; ``license`` from bare ``LICENSE``."""
        if extension and lowered_filename.endswith(extension):
            return lowered_filename[: -len(extension)]
        return lowered_filename
