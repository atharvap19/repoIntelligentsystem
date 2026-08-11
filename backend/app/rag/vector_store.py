"""ChromaDB vector store.

Each repository gets its own Chroma collection. Isolation is therefore
structural: a query against ``fastapi`` cannot return a chunk from
``langchain`` because it never searches that collection, and dropping a
repository is a collection delete rather than a metadata-filtered sweep.

Two properties matter here and are enforced rather than assumed:

* **Stable, unique IDs.** An ID is derived from the chunk's identity, not its
  position in a batch, so re-indexing the same chunk overwrites it instead of
  duplicating it.
* **Chroma-safe metadata.** Chroma rejects ``None`` and accepts only str, int,
  float and bool, so values are coerced before insert rather than blowing up
  mid-batch.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Sequence

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIRECTORY = "./chroma_db"

#: Chunk keys persisted as Chroma metadata. ``content`` is deliberately absent
#: — it is stored as the document, and duplicating it would double the index.
METADATA_FIELDS: tuple[str, ...] = (
    "repository",
    "language",
    "extension",
    "file_name",
    "relative_path",
    "chunk_index",
    "chunk_type",
    "symbol",
    "start_line",
    "end_line",
    #: Digest of the source file this chunk came from. Persisted so
    #: incremental indexing can diff without a side manifest.
    "file_hash",
)

#: Unlikely to appear inside a repository-relative path on any platform.
ID_SEPARATOR = "::"

#: Chroma collection names: 3-512 chars, alphanumeric at both ends, and only
#: alphanumerics, underscores, hyphens and single periods in between.
_VALID_COLLECTION = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,61}[a-zA-Z0-9]$")
_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

#: Key under which a collection records the repository it belongs to, so the
#: original name survives sanitisation and can be listed back.
REPOSITORY_METADATA_KEY = "repository"


def chunk_id(chunk: dict) -> str:
    """Stable, globally unique ID for a chunk.

    Includes the repository name so IDs stay unique even if collections are
    ever merged — Phase 1's ``{relative_path}_{chunk_index}`` scheme let a
    second repository's ``src/main.py`` silently overwrite the first's.
    """
    return ID_SEPARATOR.join(
        (
            str(chunk.get("repository", "")),
            str(chunk.get("relative_path", "")),
            str(chunk.get("chunk_index", 0)),
        )
    )


def collection_name_for(repository: str) -> str:
    """Map a repository name onto a legal Chroma collection name.

    Ordinary names pass through unchanged, so the collection really is called
    ``fastapi``. Anything Chroma would reject — too short, leading dot, spaces,
    an IPv4-looking name — is slugged and given a digest suffix, which keeps
    the mapping deterministic and collision-free.
    """
    name = (repository or "").strip()

    if _VALID_COLLECTION.match(name) and ".." not in name and not _IPV4.match(name):
        return name

    slug = re.sub(r"[^a-zA-Z0-9._-]", "-", name).strip("._-")[:40] or "repo"
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"repo-{slug}-{digest}"


def _coerce(value: Any) -> str | int | float | bool:
    """Force a metadata value into a type Chroma will accept."""
    if value is None:
        return ""
    if isinstance(value, bool) or isinstance(value, (int, float, str)):
        return value
    return str(value)


def metadata_for(chunk: dict) -> dict[str, str | int | float | bool]:
    """Project a chunk onto the persisted metadata fields."""
    return {field: _coerce(chunk.get(field)) for field in METADATA_FIELDS}


class VectorStore:
    """Persistence for repository embeddings, one collection per repository."""

    def __init__(
        self,
        persist_directory: str | None = None,
        distance_space: str | None = None,
    ):
        """
        Args:
            persist_directory: on-disk Chroma location.
            distance_space: HNSW space for *newly created* collections. Left
                at Chroma's default (l2) so item 9 can benchmark the switch to
                ``cosine`` — which is what nomic-embed-text is trained for —
                as a deliberate, measured change rather than a silent one.
                Changing it requires rebuilding existing collections.
        """
        self.persist_directory = persist_directory or DEFAULT_PERSIST_DIRECTORY
        self.distance_space = distance_space

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collections: dict[str, Any] = {}

    # -- collections -------------------------------------------------------

    def collection_for(self, repository: str):
        """Get or create the collection backing ``repository``."""
        if not repository:
            raise ValueError("repository name is required")

        if repository in self._collections:
            return self._collections[repository]

        metadata: dict[str, Any] = {REPOSITORY_METADATA_KEY: repository}
        if self.distance_space:
            metadata["hnsw:space"] = self.distance_space

        collection = self.client.get_or_create_collection(
            name=collection_name_for(repository),
            metadata=metadata,
        )
        self._collections[repository] = collection
        return collection

    def list_repositories(self) -> list[str]:
        """Every repository with a collection, sorted.

        Reads the repository back out of collection metadata so sanitised
        names still report their original form.
        """
        names: list[str] = []
        for entry in self.client.list_collections():
            name = entry if isinstance(entry, str) else entry.name
            try:
                collection = self.client.get_collection(name)
            except Exception:  # collection vanished between list and get
                continue
            metadata = collection.metadata or {}
            names.append(metadata.get(REPOSITORY_METADATA_KEY, name))
        return sorted(names)

    def has_repository(self, repository: str) -> bool:
        return repository in self.list_repositories()

    def delete_repository(self, repository: str) -> bool:
        """Drop a repository's collection entirely.

        Returns:
            bool: ``True`` if a collection was removed.
        """
        self._collections.pop(repository, None)
        try:
            self.client.delete_collection(name=collection_name_for(repository))
        except Exception:
            logger.info("No collection to delete for repository %r", repository)
            return False
        logger.info("Deleted collection for repository %r", repository)
        return True

    # -- writes ------------------------------------------------------------

    def add_chunk(self, repository: str, chunk: dict, embedding: Sequence[float]) -> None:
        """Store a single chunk."""
        self.add_chunks(repository, [chunk], [embedding])

    def add_chunks(
        self,
        repository: str,
        chunks: Sequence[dict],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Store a batch of chunks in one request.

        Uses ``upsert`` so re-indexing an unchanged repository is idempotent
        rather than an ID-collision error.

        Raises:
            ValueError: if the number of chunks and embeddings disagree.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings must align: {len(chunks)} != {len(embeddings)}"
            )
        if not chunks:
            return  # Chroma rejects empty batches.

        self.collection_for(repository).upsert(
            ids=[chunk_id(chunk) for chunk in chunks],
            documents=[chunk.get("content", "") for chunk in chunks],
            embeddings=list(embeddings),
            metadatas=[metadata_for(chunk) for chunk in chunks],
        )

    # -- reads -------------------------------------------------------------

    def search(
        self,
        repository: str,
        embedding: Sequence[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> dict:
        """Nearest neighbours within one repository.

        Args:
            where: optional Chroma metadata filter, applied inside the
                repository's own collection.

        Returns the raw Chroma response; ``Retriever`` shapes it into
        application objects.
        """
        return self.collection_for(repository).query(
            query_embeddings=[list(embedding)],
            n_results=max(1, top_k),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )

    def count(self, repository: str | None = None) -> int:
        """Vectors in one repository, or across all of them."""
        if repository is not None:
            return self.collection_for(repository).count()
        return sum(self.count(name) for name in self.list_repositories())

    def all_chunks(self, repository: str, include_content: bool = True) -> list[dict]:
        """Every stored chunk for a repository, as flat dicts.

        Backs the BM25 index, which needs the whole corpus rather than a
        nearest-neighbour slice. Embeddings are deliberately not fetched —
        they are the bulk of the payload and BM25 has no use for them.
        """
        include = ["metadatas", "documents"] if include_content else ["metadatas"]
        stored = self.collection_for(repository).get(include=include)

        metadatas = stored.get("metadatas") or []
        documents = stored.get("documents") or [""] * len(metadatas)

        chunks: list[dict] = []
        for metadata, document in zip(metadatas, documents):
            chunk = dict(metadata or {})
            chunk["content"] = document or ""
            chunks.append(chunk)
        return chunks

    def file_hashes(self, repository: str) -> dict[str, str]:
        """Map ``relative_path`` -> ``file_hash`` for everything indexed.

        The manifest incremental indexing diffs against. Lives in chunk
        metadata rather than a side file so it cannot drift out of sync with
        the vectors it describes.
        """
        try:
            stored = self.collection_for(repository).get(include=["metadatas"])
        except Exception:
            return {}

        hashes: dict[str, str] = {}
        for metadata in stored.get("metadatas") or []:
            metadata = metadata or {}
            path = metadata.get("relative_path")
            digest = metadata.get("file_hash")
            if path and digest:
                hashes[str(path)] = str(digest)
        return hashes

    def delete_paths(self, repository: str, relative_paths: Sequence[str]) -> int:
        """Remove every chunk belonging to the given files.

        Returns:
            int: how many vectors were removed.
        """
        paths = [p for p in relative_paths if p]
        if not paths:
            return 0

        collection = self.collection_for(repository)
        before = collection.count()
        # Chroma caps `$in` list sizes in practice; page to stay well clear.
        for start in range(0, len(paths), 200):
            page = paths[start : start + 200]
            clause = (
                {"relative_path": {"$eq": page[0]}}
                if len(page) == 1
                else {"relative_path": {"$in": page}}
            )
            collection.delete(where=clause)
        removed = before - collection.count()
        logger.info("Removed %d vectors for %d files in %r", removed, len(paths), repository)
        return removed
