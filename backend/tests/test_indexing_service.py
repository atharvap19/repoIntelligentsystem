"""Repository lifecycle tests (Phase 2, item 4).

Embeddings are faked: this exercises orchestration, not Ollama. Nothing here
touches the network or the real ./chroma_db.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.graph.store import GraphStore
from app.rag.vector_store import VectorStore
from app.services.chat_service import ChatService, UnknownRepositoryError
from app.services.indexing_service import IndexingService

from .conftest import write

DIM = 8


class FakeEmbeddingService:
    """Deterministic stand-in for Ollama."""

    def __init__(self):
        self.calls = 0
        self.batch_sizes: list[int] = []

    def generate_embeddings(self, texts):
        self.calls += 1
        self.batch_sizes.append(len(texts))
        return [[(len(t) % 10) / 10] * DIM for t in texts]

    def generate_embedding(self, text):
        return self.generate_embeddings([text])[0]


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    repo = tmp_path / "demo_repo"
    write(repo, "app/main.py", "def main():\n    return 'hello world from main'\n")
    write(repo, "app/util.py", "class Helper:\n    def run(self):\n        return 42\n")
    write(repo, "README.md", "# ignored\n")
    return repo


@pytest.fixture
def service(tmp_path: Path) -> IndexingService:
    # graph_store must be explicit: IndexingService otherwise defaults to the
    # shared ./graph_db, and a test run would write fixture repositories into
    # the real graph.
    return IndexingService(
        batch_size=2,
        repositories_root=str(tmp_path),
        embedding_service=FakeEmbeddingService(),
        vector_store=VectorStore(persist_directory=str(tmp_path / "chroma")),
        graph_store=GraphStore(path=str(tmp_path / "graph.sqlite3")),
    )


# ------------------------------------------------------------------ indexing


def test_index_repository_reports_its_work(service: IndexingService, checkout: Path):
    report = service.index_repository(checkout)

    assert report["repository"] == "demo_repo"
    assert report["files"] == 2  # README.md filtered out
    assert report["chunks"] > 0
    assert report["stored_vectors"] == report["chunks"]


def test_repository_name_defaults_to_directory_name(service: IndexingService, checkout: Path):
    service.index_repository(checkout)
    assert service.list_repositories() == ["demo_repo"]


def test_repository_name_can_be_overridden(service: IndexingService, checkout: Path):
    service.index_repository(checkout, repository="custom-name")
    assert service.list_repositories() == ["custom-name"]


def test_override_is_stamped_onto_chunk_metadata(service: IndexingService, checkout: Path):
    service.index_repository(checkout, repository="custom-name")
    stored = service.vector_store.collection_for("custom-name").peek(limit=1)
    assert stored["metadatas"][0]["repository"] == "custom-name"


def test_report_carries_timings(service: IndexingService, checkout: Path):
    report = service.index_repository(checkout)
    for key in ("parse_seconds", "chunk_seconds", "embed_seconds", "store_seconds", "total_seconds"):
        assert report[key] >= 0.0
    assert report["chunks_per_second"] >= 0.0


def test_embedding_is_batched(service: IndexingService, checkout: Path):
    service.index_repository(checkout)
    fake = service.embedding_service
    # batch_size=2, so no single call may exceed it.
    assert fake.calls >= 1
    assert max(fake.batch_sizes) <= 2


def test_indexing_is_idempotent(service: IndexingService, checkout: Path):
    first = service.index_repository(checkout)
    second = service.index_repository(checkout)
    assert first["stored_vectors"] == second["stored_vectors"]


# ------------------------------------------------------------------ isolation


def test_two_repositories_stay_separate(service: IndexingService, checkout: Path, tmp_path: Path):
    other = tmp_path / "other_repo"
    write(other, "app/main.py", "def other():\n    return 'a completely different body'\n")

    service.index_repository(checkout)
    service.index_repository(other)

    assert service.list_repositories() == ["demo_repo", "other_repo"]
    assert service.vector_store.count("demo_repo") > 0
    assert service.vector_store.count("other_repo") > 0


def test_identical_paths_across_repositories_do_not_overwrite(
    service: IndexingService, tmp_path: Path
):
    for name in ("alpha", "beta"):
        repo = tmp_path / name
        write(repo, "src/main.py", f"def {name}():\n    return '{name} implementation body'\n")
        service.index_repository(repo)

    alpha = service.vector_store.collection_for("alpha").peek(limit=1)["documents"][0]
    beta = service.vector_store.collection_for("beta").peek(limit=1)["documents"][0]
    assert "alpha" in alpha
    assert "beta" in beta


# ------------------------------------------------------------------ deletion


def test_delete_repository(service: IndexingService, checkout: Path):
    service.index_repository(checkout)
    assert service.delete_repository("demo_repo") == {"repository": "demo_repo", "deleted": True}
    assert service.list_repositories() == []


def test_delete_unknown_repository_is_not_an_error(service: IndexingService):
    assert service.delete_repository("ghost")["deleted"] is False


def test_delete_leaves_other_repositories_intact(
    service: IndexingService, checkout: Path, tmp_path: Path
):
    other = tmp_path / "other_repo"
    write(other, "a.py", "def other():\n    return 'body text here for size'\n")
    service.index_repository(checkout)
    service.index_repository(other)

    service.delete_repository("demo_repo")
    assert service.list_repositories() == ["other_repo"]


# ----------------------------------------------------------------- reindexing


def test_reindex_rebuilds_from_the_checkout(service: IndexingService, checkout: Path):
    service.index_repository(checkout)
    report = service.reindex_repository("demo_repo", repository_path=checkout)
    assert report["chunks"] > 0
    assert service.vector_store.count("demo_repo") == report["chunks"]


def test_reindex_drops_chunks_for_deleted_files(service: IndexingService, checkout: Path):
    service.index_repository(checkout)
    before = service.vector_store.count("demo_repo")

    (checkout / "app" / "util.py").unlink()
    service.reindex_repository("demo_repo", repository_path=checkout)

    # A metadata-preserving update would have left the stale chunks behind.
    assert service.vector_store.count("demo_repo") < before


def test_reindex_finds_the_checkout_under_the_repositories_root(
    tmp_path: Path,
):
    root = tmp_path / "repositories"
    write(root / "demo", "a.py", "def a():\n    return 'enough body text to survive'\n")
    service = IndexingService(
        repositories_root=str(root),
        embedding_service=FakeEmbeddingService(),
        vector_store=VectorStore(persist_directory=str(tmp_path / "chroma")),
        graph_store=GraphStore(path=str(tmp_path / "graph.sqlite3")),
    )
    report = service.reindex_repository("demo")
    assert report["repository"] == "demo"


def test_reindex_without_a_checkout_raises(service: IndexingService):
    with pytest.raises(FileNotFoundError):
        service.reindex_repository("missing")


# ------------------------------------------------- repository resolution (chat)


class _StubRetriever:
    def __init__(self, store):
        self.vector_store = store


def chat_service_for(store: VectorStore) -> ChatService:
    return ChatService(
        embedding_service=FakeEmbeddingService(),
        retriever=_StubRetriever(store),
        llm_service=object(),
    )


def test_single_repository_is_resolved_implicitly(service: IndexingService, checkout: Path):
    service.index_repository(checkout)
    chat = chat_service_for(service.vector_store)
    assert chat.resolve_repository(None) == "demo_repo"


def test_explicit_repository_is_honoured(service: IndexingService, checkout: Path):
    service.index_repository(checkout)
    chat = chat_service_for(service.vector_store)
    assert chat.resolve_repository("demo_repo") == "demo_repo"


def test_unknown_repository_is_rejected_with_available_names(
    service: IndexingService, checkout: Path
):
    service.index_repository(checkout)
    chat = chat_service_for(service.vector_store)
    with pytest.raises(UnknownRepositoryError, match="demo_repo"):
        chat.resolve_repository("nope")


def test_ambiguous_repository_is_rejected(service: IndexingService, checkout: Path, tmp_path: Path):
    other = tmp_path / "other_repo"
    write(other, "a.py", "def other():\n    return 'body text here for size'\n")
    service.index_repository(checkout)
    service.index_repository(other)

    chat = chat_service_for(service.vector_store)
    with pytest.raises(UnknownRepositoryError, match="required"):
        chat.resolve_repository(None)


def test_no_repositories_gives_an_actionable_error(tmp_path: Path):
    store = VectorStore(persist_directory=str(tmp_path / "chroma"))
    with pytest.raises(UnknownRepositoryError, match="No repositories"):
        chat_service_for(store).resolve_repository(None)


# ------------------------------------------------------ item 8: incremental


def test_update_falls_back_to_full_build_when_never_indexed(
    service: IndexingService, checkout: Path
):
    report = service.update_repository(checkout)
    assert report["mode"] == "full"
    assert report["chunks"] > 0


def test_update_skips_unchanged_files(service: IndexingService, checkout: Path):
    service.index_repository(checkout)
    service.embedding_service.calls = 0

    report = service.update_repository(checkout)

    assert report["mode"] == "incremental"
    assert report["files_unchanged"] == 2
    assert report["files_added"] == 0
    assert report["files_changed"] == 0
    # The whole point: nothing unchanged is re-embedded.
    assert service.embedding_service.calls == 0


def test_update_reprocesses_only_the_changed_file(service: IndexingService, checkout: Path):
    service.index_repository(checkout)
    (checkout / "app" / "main.py").write_text(
        "def main():\n    return 'a completely rewritten body for main'\n",
        encoding="utf-8", newline="",
    )

    report = service.update_repository(checkout)

    assert report["files_changed"] == 1
    assert report["files_unchanged"] == 1
    assert report["vectors_removed"] >= 1


def test_update_adds_new_files(service: IndexingService, checkout: Path):
    service.index_repository(checkout)
    write(checkout, "app/extra.py", "def extra():\n    return 'brand new file body'\n")

    report = service.update_repository(checkout)

    assert report["files_added"] == 1
    assert report["files_changed"] == 0


def test_update_removes_vectors_for_deleted_files(service: IndexingService, checkout: Path):
    service.index_repository(checkout)
    before = service.vector_store.count("demo_repo")

    (checkout / "app" / "util.py").unlink()
    report = service.update_repository(checkout)

    assert report["files_removed"] == 1
    assert service.vector_store.count("demo_repo") < before


def test_shrinking_a_file_leaves_no_orphan_chunks(service: IndexingService, tmp_path: Path):
    repo = tmp_path / "shrink"
    long_body = "".join(f"def f{i}():\n    return {i} * 1000\n\n" for i in range(40))
    write(repo, "big.py", long_body)
    service.index_repository(repo)
    before = service.vector_store.count("shrink")

    # A plain upsert would leave the high-index chunks behind.
    write(repo, "big.py", "def f0():\n    return 'only one function remains now'\n")
    service.update_repository(repo)

    after = service.vector_store.count("shrink")
    assert after < before
    stored = service.vector_store.all_chunks("shrink")
    assert all(c["chunk_index"] == 0 for c in stored)


def test_update_is_idempotent(service: IndexingService, checkout: Path):
    service.index_repository(checkout)
    first = service.update_repository(checkout)
    second = service.update_repository(checkout)
    assert first["stored_vectors"] == second["stored_vectors"]


def test_file_hash_is_persisted(service: IndexingService, checkout: Path):
    service.index_repository(checkout)
    hashes = service.vector_store.file_hashes("demo_repo")
    assert set(hashes) == {"app/main.py", "app/util.py"}
    assert all(len(h) == 32 for h in hashes.values())


def test_delete_paths_returns_removed_count(service: IndexingService, checkout: Path):
    service.index_repository(checkout)
    removed = service.vector_store.delete_paths("demo_repo", ["app/main.py"])
    assert removed >= 1
    assert "app/main.py" not in service.vector_store.file_hashes("demo_repo")


def test_delete_paths_with_no_paths_is_a_noop(service: IndexingService, checkout: Path):
    service.index_repository(checkout)
    assert service.vector_store.delete_paths("demo_repo", []) == 0


def test_keyword_cache_is_invalidated_on_write(service: IndexingService, checkout: Path):
    class Spy:
        def __init__(self):
            self.calls = []

        def invalidate(self, repository=None):
            self.calls.append(repository)

    service.keyword_cache = Spy()
    service.index_repository(checkout)
    assert service.keyword_cache.calls == ["demo_repo"]
