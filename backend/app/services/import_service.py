"""Import a GitHub URL and turn it into a knowledge graph you can talk to.

One call replaces the old two-step clone-then-index. The work runs on a
background thread because embedding a large repository takes minutes. The graph
is built first, in seconds, so the UI can draw it — and the agent can answer
from it — while code search is still being indexed.

Jobs live in process memory. Repoint is a local single-user tool (see
``app/api/indexing.py``); a real queue belongs here before it serves more.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from app.services.github_service import GitHubService, parse_github_url
from app.services.indexing_service import IndexingService

logger = logging.getLogger(__name__)

STATE_CLONING = "cloning"
STATE_BUILDING_GRAPH = "building_graph"
STATE_INDEXING_CODE = "indexing_code"
STATE_READY = "ready"
STATE_ERROR = "error"

FINISHED_STATES = frozenset({STATE_READY, STATE_ERROR})


@dataclass
class ImportJob:
    """Where one repository is on its way from URL to graph."""

    repository: str
    url: str = ""
    state: str = STATE_CLONING
    message: str = ""
    #: The knowledge graph exists and can be drawn and asked about.
    graph_ready: bool = False
    #: Code chunks are embedded, so answers can quote source.
    search_ready: bool = False
    #: Chunk counts while embedding.
    progress_done: int = 0
    progress_total: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def finished(self) -> bool:
        return self.state in FINISHED_STATES

    def to_dict(self) -> dict:
        return asdict(self)


class ImportService:
    """Starts import jobs and reports on them."""

    def __init__(
        self,
        indexing: IndexingService,
        clone: Callable[[str], dict] | None = None,
        on_indexed: Callable[[str], None] | None = None,
        run_async: bool = True,
    ):
        self.indexing = indexing
        self.clone = clone or (lambda url: GitHubService().clone_repository(url))
        #: Called after a successful index, to drop query-side caches.
        self.on_indexed = on_indexed
        self.run_async = run_async
        self._jobs: dict[str, ImportJob] = {}
        self._lock = threading.Lock()
        #: Imports clone in parallel but index one at a time: they share one
        #: vector store and one local embedding model, so running two at once
        #: only makes both slower.
        self._indexing_lock = threading.Lock()

    # -- queries -----------------------------------------------------------

    def job(self, repository: str) -> ImportJob | None:
        with self._lock:
            return self._jobs.get(repository)

    def jobs(self) -> list[ImportJob]:
        with self._lock:
            return list(self._jobs.values())

    def graph_exists(self, repository: str) -> bool:
        store = self.indexing.graph_store
        return store is not None and store.has_repository(repository)

    def search_exists(self, repository: str) -> bool:
        return self.indexing.vector_store.count(repository) > 0

    def status(self, repository: str) -> dict | None:
        """The job if one ran this session, else what is already on disk."""
        job = self.job(repository)
        if job is not None:
            return job.to_dict()
        if not self.graph_exists(repository):
            return None
        return ImportJob(
            repository=repository,
            state=STATE_READY,
            graph_ready=True,
            search_ready=self.search_exists(repository),
            finished_at=time.time(),
        ).to_dict()

    # -- commands ----------------------------------------------------------

    def start(self, url: str) -> ImportJob:
        """Validate the URL and begin importing.

        Raises:
            InvalidRepositoryUrl: before any work starts.
        """
        _, repository = parse_github_url(url)

        with self._lock:
            existing = self._jobs.get(repository)
            # Asking twice for a repository already on its way is not a
            # second import.
            if existing is not None and not existing.finished:
                return existing
            job = ImportJob(repository=repository, url=url.strip())
            self._jobs[repository] = job

        if self.run_async:
            threading.Thread(
                target=self._run, args=(job,), name=f"import-{repository}", daemon=True
            ).start()
        else:
            self._run(job)
        return job

    def _run(self, job: ImportJob) -> None:
        try:
            result = self.clone(job.url)
            if result.get("status") != "success":
                raise RuntimeError(result.get("message") or "Clone failed.")
            path = Path(result["path"])

            if self.graph_exists(job.repository) and self.search_exists(job.repository):
                job.message = "Already imported."
                job.graph_ready = job.search_ready = True
                job.state = STATE_READY
                return

            if not self._indexing_lock.acquire(blocking=False):
                job.message = "Waiting for another import to finish."
                self._indexing_lock.acquire()
            try:
                job.message = ""
                self.indexing.index_repository(
                    path,
                    repository=job.repository,
                    progress=lambda phase, done, total: self._progress(job, phase, done, total),
                )
            finally:
                self._indexing_lock.release()
            if self.on_indexed:
                self.on_indexed(job.repository)

            job.graph_ready = self.graph_exists(job.repository)
            job.search_ready = True
            job.state = STATE_READY
            if not job.graph_ready:
                job.message = "Code search is ready, but the graph could not be built."
        except Exception as exc:  # noqa: BLE001 - the job reports every failure
            logger.exception("Import of %r failed", job.repository)
            job.graph_ready = self.graph_exists(job.repository)
            detail = str(exc) or type(exc).__name__
            job.message = (
                f"Code search indexing failed ({detail}). The graph is available; "
                "answers will rely on it alone."
                if job.graph_ready
                else detail
            )
            job.state = STATE_ERROR
        finally:
            job.finished_at = time.time()

    def _progress(self, job: ImportJob, phase: str, done: int, total: int) -> None:
        if phase in ("parsing", "graph"):
            job.state = STATE_BUILDING_GRAPH
        elif phase == "embedding":
            job.graph_ready = self.graph_exists(job.repository)
            job.state = STATE_INDEXING_CODE
            job.progress_done, job.progress_total = done, total
