"""Indexing service.

Owns a repository's lifecycle in the vector store: build it, drop it, rebuild
it. Embeddings are produced and stored in batches so a large repository's
vectors never accumulate in memory.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.graph.builder import build_repository_graph
from app.graph.git_history import extract_history
from app.graph.store import GraphStore
from app.rag.chunker import RepositoryChunker
from app.rag.parser import RepositoryParser
from app.rag.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

DEFAULT_REPOSITORIES_ROOT = "repositories"
DEFAULT_BATCH_SIZE = 32

#: Called as ``progress(phase, done, total)``. Phases, in order: ``parsing``,
#: ``graph``, ``embedding`` (repeated per batch with chunk counts).
ProgressCallback = Callable[[str, int, int], None]


@dataclass
class IndexReport:
    """Outcome and timings of one indexing run."""

    repository: str = ""
    mode: str = "full"
    files: int = 0
    chunks: int = 0
    stored_vectors: int = 0
    #: Incremental accounting; all zero on a full build.
    files_added: int = 0
    files_changed: int = 0
    files_removed: int = 0
    files_unchanged: int = 0
    vectors_removed: int = 0
    parse_seconds: float = 0.0
    chunk_seconds: float = 0.0
    embed_seconds: float = 0.0
    store_seconds: float = 0.0
    graph_seconds: float = 0.0
    history_seconds: float = 0.0
    total_seconds: float = 0.0
    chunk_types: dict = field(default_factory=dict)
    #: Phase 3 structural/temporal layer; zero when graph building is off.
    graph_nodes: int = 0
    graph_edges: int = 0
    graph_symbols: int = 0
    commits: int = 0

    @property
    def chunks_per_second(self) -> float:
        return self.chunks / self.total_seconds if self.total_seconds else 0.0

    @property
    def embeddings_per_second(self) -> float:
        return self.chunks / self.embed_seconds if self.embed_seconds else 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["chunks_per_second"] = round(self.chunks_per_second, 1)
        data["embeddings_per_second"] = round(self.embeddings_per_second, 1)
        return data


class IndexingService:
    """Builds and maintains a repository's vector index.

    Pipeline::

        repository path
            -> RepositoryParser      (source filtering)
            -> RepositoryChunker     (AST chunking)
            -> EmbeddingService      (batched)
            -> VectorStore           (per-repository collection)
    """

    def __init__(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        repositories_root: str | None = None,
        parser: RepositoryParser | None = None,
        chunker: RepositoryChunker | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
        keyword_cache: Any | None = None,
        graph_store: GraphStore | None = None,
        build_graph: bool = True,
        max_commits: int = 1500,
    ):
        self.batch_size = max(1, batch_size)
        self.repositories_root = Path(repositories_root or DEFAULT_REPOSITORIES_ROOT)

        self.parser = parser or RepositoryParser()
        self.chunker = chunker or RepositoryChunker()
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()
        #: Phase 3. Graph building costs ~2s against ~230s of embedding, so it
        #: is on by default; tests that only exercise retrieval can turn it off.
        self.graph_store = graph_store if graph_store is not None else GraphStore()
        self.build_graph = build_graph
        self.max_commits = max_commits
        #: Optional object with an ``invalidate(repository)`` method — the
        #: keyword retriever, when indexing and querying share a process.
        self.keyword_cache = keyword_cache

    # -- lifecycle ---------------------------------------------------------

    def index_repository(
        self,
        repository_path: str | Path,
        repository: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> dict:
        """Index a repository into its own collection.

        The graph is built before embedding: it takes seconds where embedding
        takes minutes, so a caller watching ``progress`` can show the graph
        long before code search is ready.

        Args:
            repository_path: local path to the checkout.
            repository: collection name. Defaults to the directory name, so
                nothing about the repository is hardcoded.
            progress: optional :data:`ProgressCallback`.

        Returns:
            dict: an :class:`IndexReport`.
        """
        path = Path(repository_path)
        name = repository or path.name
        report = IndexReport(repository=name)
        started = time.perf_counter()
        notify = progress or (lambda phase, done, total: None)

        logger.info("Indexing %r from %s", name, path)

        notify("parsing", 0, 0)
        mark = time.perf_counter()
        files = self.parser.parse_repository(path)
        report.parse_seconds = time.perf_counter() - mark
        report.files = len(files)
        logger.info("Parsed %d files in %.2fs", report.files, report.parse_seconds)

        notify("graph", 0, 0)
        self._build_graph(name, path, files, report)

        mark = time.perf_counter()
        chunks = self.chunker.chunk_repository(files)
        report.chunk_seconds = time.perf_counter() - mark
        report.chunks = len(chunks)
        report.chunk_types = dict(self.chunker.last_stats.by_type)
        logger.info("Created %d chunks in %.2fs", report.chunks, report.chunk_seconds)

        # Stamp the collection name onto every chunk so metadata, IDs and the
        # collection all agree even when `repository` overrides the directory.
        for chunk in chunks:
            chunk["repository"] = name

        self._embed_and_store(name, chunks, report, notify)

        report.total_seconds = time.perf_counter() - started
        report.stored_vectors = self.vector_store.count(name)
        self._invalidate_caches(name)
        logger.info(
            "Indexed %r: %d chunks in %.2fs (%.1f chunks/s)",
            name,
            report.chunks,
            report.total_seconds,
            report.chunks_per_second,
        )
        return report.to_dict()

    def update_repository(
        self,
        repository_path: str | Path,
        repository: str | None = None,
    ) -> dict:
        """Re-index only what changed since the last run.

        Diffs the checkout's file hashes against the hashes already stored in
        the collection:

        * unchanged files are skipped entirely — no parse cost beyond reading
          them, and crucially no embedding cost;
        * changed files have their old chunks deleted before the new ones are
          written, because a file that shrinks would otherwise leave orphaned
          chunks behind that ``upsert`` never touches;
        * deleted files have their chunks removed.

        Falls back to a full build when the repository has never been indexed.
        """
        path = Path(repository_path)
        name = repository or path.name

        stored_hashes = self.vector_store.file_hashes(name)
        if not stored_hashes:
            logger.info("No prior index for %r; running a full build", name)
            return self.index_repository(path, repository=name)

        report = IndexReport(repository=name, mode="incremental")
        started = time.perf_counter()

        mark = time.perf_counter()
        files = self.parser.parse_repository(path)
        report.parse_seconds = time.perf_counter() - mark
        report.files = len(files)

        current = {f["relative_path"]: f.get("file_hash", "") for f in files}

        added = [p for p in current if p not in stored_hashes]
        changed = [p for p in current if p in stored_hashes and current[p] != stored_hashes[p]]
        removed = [p for p in stored_hashes if p not in current]

        report.files_added = len(added)
        report.files_changed = len(changed)
        report.files_removed = len(removed)
        report.files_unchanged = len(current) - len(added) - len(changed)

        logger.info(
            "Diff for %r: +%d ~%d -%d (=%d unchanged)",
            name, len(added), len(changed), len(removed), report.files_unchanged,
        )

        # Stale chunks must go before the rewrite: a shrunken file leaves
        # high-index chunks that upsert would never overwrite.
        stale = changed + removed
        if stale:
            report.vectors_removed = self.vector_store.delete_paths(name, stale)

        dirty = set(added) | set(changed)
        if dirty:
            mark = time.perf_counter()
            chunks = self.chunker.chunk_repository([f for f in files if f["relative_path"] in dirty])
            report.chunk_seconds = time.perf_counter() - mark
            report.chunks = len(chunks)
            report.chunk_types = dict(self.chunker.last_stats.by_type)

            for chunk in chunks:
                chunk["repository"] = name

            self._embed_and_store(name, chunks, report)

        report.total_seconds = time.perf_counter() - started
        report.stored_vectors = self.vector_store.count(name)
        self._invalidate_caches(name)

        logger.info(
            "Updated %r in %.2fs: %d chunks re-embedded, %d vectors removed",
            name, report.total_seconds, report.chunks, report.vectors_removed,
        )
        return report.to_dict()

    def delete_repository(self, repository: str) -> dict:
        """Drop a repository's collection and its graph."""
        deleted = self.vector_store.delete_repository(repository)
        if self.graph_store is not None:
            self.graph_store.delete_repository(repository)
        self._invalidate_caches(repository)
        return {"repository": repository, "deleted": deleted}

    def reindex_repository(
        self,
        repository: str,
        repository_path: str | Path | None = None,
    ) -> dict:
        """Rebuild a repository from scratch.

        Drops the collection first so chunks belonging to deleted or renamed
        files cannot survive the rebuild. Incremental, hash-based updates
        arrive in item 8; this is the correct-but-complete path.
        """
        path = Path(repository_path) if repository_path else self.repositories_root / repository
        if not path.exists():
            raise FileNotFoundError(f"Repository checkout not found: {path}")

        self.vector_store.delete_repository(repository)
        return self.index_repository(path, repository=repository)

    def list_repositories(self) -> list[str]:
        return self.vector_store.list_repositories()

    # -- internals ---------------------------------------------------------

    def _build_graph(
        self,
        repository: str,
        path: Path,
        files: list[dict],
        report: IndexReport,
    ) -> None:
        """Build and persist the structural graph and the git timeline.

        Reuses the already-parsed file list rather than re-reading the
        checkout, so the graph and the vector index always describe the same
        corpus. Failures are logged and swallowed: a broken graph must not
        cost the user a completed embedding run.
        """
        if not self.build_graph or self.graph_store is None:
            return

        try:
            mark = time.perf_counter()
            graph, graph_stats = build_repository_graph(repository, files)
            self.graph_store.replace_repository(repository, graph.nodes, graph.edges)
            report.graph_seconds = time.perf_counter() - mark
            report.graph_nodes = len(graph.nodes)
            report.graph_edges = len(graph.edges)
            report.graph_symbols = graph_stats.symbols
        except Exception:
            logger.exception("Graph build failed for %r; index is still valid", repository)
            return

        try:
            mark = time.perf_counter()
            history = extract_history(path, max_commits=self.max_commits)
            self.graph_store.replace_history(
                repository, history.commits, history.commit_files, history.refs
            )
            report.history_seconds = time.perf_counter() - mark
            report.commits = history.stats.commits
        except Exception:
            logger.exception("History extraction failed for %r", repository)

    def _invalidate_caches(self, repository: str) -> None:
        """Tell any attached keyword index that the corpus moved underneath it.

        The BM25 index is an in-memory mirror of the collection; leaving it
        stale after a write would silently serve deleted chunks.
        """
        invalidate = getattr(self.keyword_cache, "invalidate", None)
        if callable(invalidate):
            invalidate(repository)

    def _embed_and_store(
        self,
        repository: str,
        chunks: list[dict],
        report: IndexReport,
        progress: ProgressCallback | None = None,
    ) -> None:
        """Embed and persist in batches, discarding each batch as it lands."""
        if progress:
            progress("embedding", 0, len(chunks))
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]

            mark = time.perf_counter()
            embeddings = self.embedding_service.generate_embeddings(
                [chunk["content"] for chunk in batch]
            )
            report.embed_seconds += time.perf_counter() - mark

            mark = time.perf_counter()
            self.vector_store.add_chunks(repository, batch, embeddings)
            report.store_seconds += time.perf_counter() - mark

            processed = min(start + self.batch_size, len(chunks))
            logger.debug("Embedded %d/%d chunks", processed, len(chunks))
            if progress:
                progress("embedding", processed, len(chunks))
