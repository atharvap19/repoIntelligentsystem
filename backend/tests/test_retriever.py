"""Retriever output-shape tests (Phase 2, item 3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.retriever import RESULT_FIELDS, Retriever, relevance_score
from app.rag.vector_store import METADATA_FIELDS, VectorStore

from .test_vector_store import DIM, embedding, make_chunk


REPO = "fastapi"


@pytest.fixture
def retriever(tmp_path: Path) -> Retriever:
    store = VectorStore(persist_directory=str(tmp_path / "chroma"))
    chunks = [
        make_chunk(chunk_index=0, symbol="APIRouter.include_router", content="include_router body"),
        make_chunk(chunk_index=1, symbol="APIRouter.add_api_route", content="add_api_route body"),
        make_chunk(chunk_index=2, symbol="serialize_response", content="serialize body"),
    ]
    store.add_chunks(REPO, chunks, [embedding(0.1), embedding(0.2), embedding(0.3)])
    return Retriever(vector_store=store)


# ------------------------------------------------------------------- shaping


def test_results_carry_every_declared_field(retriever: Retriever):
    for result in retriever.retrieve(embedding(0.1), REPO, top_k=3):
        assert set(result) == set(RESULT_FIELDS)


def test_metadata_survives_the_round_trip(retriever: Retriever):
    best = retriever.retrieve(embedding(0.1), REPO, top_k=1)[0]

    assert best["symbol"] == "APIRouter.include_router"
    assert best["chunk_type"] == "method"
    assert best["relative_path"] == "fastapi/routing.py"
    assert best["start_line"] == 100
    assert best["end_line"] == 150


def test_content_is_returned(retriever: Retriever):
    assert retriever.retrieve(embedding(0.1), REPO, top_k=1)[0]["content"] == "include_router body"


def test_no_raw_chroma_shape_leaks(retriever: Retriever):
    results = retriever.retrieve(embedding(0.1), REPO, top_k=3)
    assert isinstance(results, list)
    # Chroma's parallel-array keys must not appear in application objects.
    for result in results:
        assert "documents" not in result
        assert "metadatas" not in result
        assert "ids" not in result


def test_top_k_is_respected(retriever: Retriever):
    assert len(retriever.retrieve(embedding(0.1), REPO, top_k=2)) == 2


def test_results_are_ordered_best_first(retriever: Retriever):
    results = retriever.retrieve(embedding(0.1), REPO, top_k=3)
    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_collection_returns_empty_list(tmp_path: Path):
    store = VectorStore(persist_directory=str(tmp_path / "c"))
    assert Retriever(vector_store=store).retrieve([0.0] * DIM, "empty", top_k=5) == []


# --------------------------------------------------------------------- score


def test_score_decreases_as_distance_grows():
    assert relevance_score(0.0) > relevance_score(0.5) > relevance_score(2.0)


def test_score_is_bounded():
    assert relevance_score(0.0) == 1.0
    assert 0.0 < relevance_score(1000.0) < 0.01


def test_negative_distance_is_clamped():
    assert relevance_score(-1.0) == 0.0


def test_result_fields_superset_of_metadata_fields():
    assert set(METADATA_FIELDS) <= set(RESULT_FIELDS)
