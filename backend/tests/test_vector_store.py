"""Vector store: metadata, IDs and per-repository collections (items 3-4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.chunker import CHUNK_METHOD
from app.rag.vector_store import (
    ID_SEPARATOR,
    METADATA_FIELDS,
    VectorStore,
    chunk_id,
    collection_name_for,
    metadata_for,
)

DIM = 8


def make_chunk(**overrides) -> dict:
    chunk = {
        "repository": "fastapi",
        "language": "python",
        "extension": ".py",
        "file_name": "routing.py",
        "relative_path": "fastapi/routing.py",
        "chunk_index": 3,
        "chunk_type": CHUNK_METHOD,
        "symbol": "APIRouter.include_router",
        "start_line": 100,
        "end_line": 150,
        "content": "def include_router(self):\n    ...",
    }
    chunk.update(overrides)
    return chunk


def embedding(seed: float = 0.1) -> list[float]:
    return [seed] * DIM


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    """A throwaway store — never the real ./chroma_db."""
    return VectorStore(persist_directory=str(tmp_path / "chroma"))


# ---------------------------------------------------------------------- ids


def test_chunk_id_is_scoped_by_repository():
    a = chunk_id(make_chunk(repository="alpha"))
    b = chunk_id(make_chunk(repository="beta"))
    assert a != b
    assert a.startswith("alpha" + ID_SEPARATOR)


def test_chunk_id_is_stable_across_calls():
    assert chunk_id(make_chunk()) == chunk_id(make_chunk())


def test_chunk_id_distinguishes_chunk_index():
    assert chunk_id(make_chunk(chunk_index=1)) != chunk_id(make_chunk(chunk_index=2))


# -------------------------------------------------------------- collections


@pytest.mark.parametrize("name", ["fastapi", "langchain", "my_project", "repo-1.2"])
def test_ordinary_names_are_used_verbatim(name: str):
    assert collection_name_for(name) == name


@pytest.mark.parametrize(
    "name",
    ["ab", "", "  ", ".hidden", "has spaces", "sla/sh", "192.168.0.1", "dots..double"],
)
def test_illegal_names_are_sanitised(name: str):
    result = collection_name_for(name)
    assert 3 <= len(result) <= 63
    assert result[0].isalnum() and result[-1].isalnum()
    assert all(c.isalnum() or c in "._-" for c in result)


def test_sanitisation_is_deterministic_and_distinct():
    assert collection_name_for("has spaces") == collection_name_for("has spaces")
    assert collection_name_for("a b") != collection_name_for("a-b")


def test_each_repository_gets_its_own_collection(store: VectorStore):
    store.add_chunks("alpha", [make_chunk(repository="alpha")], [embedding(0.1)])
    store.add_chunks("beta", [make_chunk(repository="beta")], [embedding(0.2)])

    assert store.list_repositories() == ["alpha", "beta"]
    assert store.count("alpha") == 1
    assert store.count("beta") == 1


def test_a_query_cannot_see_another_repository(store: VectorStore):
    store.add_chunks("alpha", [make_chunk(repository="alpha", content="alpha body")], [embedding(0.1)])
    store.add_chunks("beta", [make_chunk(repository="beta", content="beta body")], [embedding(0.1)])

    documents = store.search("alpha", embedding(0.1), top_k=10)["documents"][0]
    assert documents == ["alpha body"]


def test_same_path_in_two_repositories_does_not_collide(store: VectorStore):
    a = make_chunk(repository="alpha", relative_path="src/main.py", content="alpha")
    b = make_chunk(repository="beta", relative_path="src/main.py", content="beta")

    store.add_chunks("alpha", [a], [embedding(0.1)])
    store.add_chunks("beta", [b], [embedding(0.2)])

    assert store.count("alpha") == 1
    assert store.count("beta") == 1
    assert store.count() == 2


def test_repository_name_is_recoverable_after_sanitisation(store: VectorStore):
    store.add_chunks("has spaces", [make_chunk(repository="has spaces")], [embedding()])
    assert "has spaces" in store.list_repositories()


def test_delete_repository_removes_only_that_collection(store: VectorStore):
    store.add_chunks("alpha", [make_chunk(repository="alpha")], [embedding(0.1)])
    store.add_chunks("beta", [make_chunk(repository="beta")], [embedding(0.2)])

    assert store.delete_repository("alpha") is True
    assert store.list_repositories() == ["beta"]
    assert store.count("beta") == 1


def test_deleting_an_unknown_repository_is_reported_not_raised(store: VectorStore):
    assert store.delete_repository("never-indexed") is False


def test_delete_then_reuse_starts_empty(store: VectorStore):
    store.add_chunks("alpha", [make_chunk(repository="alpha")], [embedding()])
    store.delete_repository("alpha")
    assert store.count("alpha") == 0


def test_has_repository(store: VectorStore):
    store.add_chunks("alpha", [make_chunk(repository="alpha")], [embedding()])
    assert store.has_repository("alpha")
    assert not store.has_repository("beta")


def test_count_without_argument_totals_every_repository(store: VectorStore):
    store.add_chunks("alpha", [make_chunk(repository="alpha", chunk_index=i) for i in range(3)],
                     [embedding(0.1)] * 3)
    store.add_chunks("beta", [make_chunk(repository="beta")], [embedding(0.2)])
    assert store.count() == 4


def test_empty_repository_name_is_rejected(store: VectorStore):
    with pytest.raises(ValueError):
        store.collection_for("")


# ----------------------------------------------------------------- metadata


def test_metadata_includes_every_declared_field():
    assert set(metadata_for(make_chunk())) == set(METADATA_FIELDS)


def test_metadata_excludes_content():
    # Content lives in the document; duplicating it would double the index.
    assert "content" not in metadata_for(make_chunk())


def test_ast_fields_are_persisted(store: VectorStore):
    store.add_chunks("fastapi", [make_chunk()], [embedding()])
    stored = store.collection_for("fastapi").peek(limit=1)["metadatas"][0]

    assert stored["chunk_type"] == CHUNK_METHOD
    assert stored["symbol"] == "APIRouter.include_router"
    assert stored["start_line"] == 100
    assert stored["end_line"] == 150


def test_none_is_coerced_rather_than_rejected(store: VectorStore):
    # Chroma raises on None; the store must normalise before insert.
    store.add_chunks("fastapi", [make_chunk(symbol=None)], [embedding()])
    assert store.collection_for("fastapi").peek(limit=1)["metadatas"][0]["symbol"] == ""


def test_missing_keys_do_not_break_insert(store: VectorStore):
    store.add_chunks("r", [{"repository": "r", "relative_path": "a.py", "chunk_index": 0}], [embedding()])
    assert store.count("r") == 1


@pytest.mark.parametrize("field", METADATA_FIELDS)
def test_every_metadata_value_is_chroma_compatible(field: str):
    value = metadata_for(make_chunk())[field]
    assert isinstance(value, (str, int, float, bool))


# ------------------------------------------------------------------- writes


def test_reindexing_the_same_chunk_upserts(store: VectorStore):
    store.add_chunks("fastapi", [make_chunk(content="first")], [embedding()])
    store.add_chunks("fastapi", [make_chunk(content="second")], [embedding()])

    assert store.count("fastapi") == 1
    assert store.collection_for("fastapi").peek(limit=1)["documents"][0] == "second"


def test_mismatched_lengths_raise(store: VectorStore):
    with pytest.raises(ValueError):
        store.add_chunks("fastapi", [make_chunk()], [])


def test_empty_batch_is_a_noop(store: VectorStore):
    store.add_chunks("fastapi", [], [])
    assert store.count("fastapi") == 0


def test_add_chunk_delegates_to_batch(store: VectorStore):
    store.add_chunk("fastapi", make_chunk(), embedding())
    assert store.count("fastapi") == 1


# -------------------------------------------------------------------- reads


def test_search_returns_requested_count(store: VectorStore):
    chunks = [make_chunk(chunk_index=i, content=f"body {i}") for i in range(5)]
    store.add_chunks("fastapi", chunks, [embedding(i / 10) for i in range(5)])

    response = store.search("fastapi", embedding(0.0), top_k=3)
    assert len(response["documents"][0]) == 3


def test_search_on_empty_collection_is_safe(store: VectorStore):
    response = store.search("fastapi", embedding(), top_k=5)
    assert (response.get("documents") or [[]])[0] in ([], None)
