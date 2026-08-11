"""Filtering, BM25, hybrid fusion and reranking (Phase 2, items 5-7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.bm25 import BM25Index
from app.rag.filters import FILTERABLE_FIELDS, build_where, matches
from app.rag.reranker import HeuristicReranker, IdentityReranker, path_prior
from app.rag.retriever import (
    HybridRetriever,
    KeywordRetriever,
    Retriever,
    VectorRetriever,
    normalise,
)
from app.rag.tokenize import chunk_terms, split_identifier, tokenize, tokenize_path
from app.rag.vector_store import VectorStore

DIM = 8
REPO = "fastapi"


def chunk(path, symbol, content, chunk_type="function", index=0, language="python", extension=".py"):
    return {
        "repository": REPO,
        "language": language,
        "extension": extension,
        "file_name": path.split("/")[-1],
        "relative_path": path,
        "file_hash": "deadbeef",
        "chunk_index": index,
        "chunk_type": chunk_type,
        "symbol": symbol,
        "start_line": 1,
        "end_line": 10,
        "content": content,
    }


CORPUS = [
    chunk("fastapi/routing.py", "APIRouter.include_router",
          "def include_router(self, router, prefix=''): self.routes.extend(router.routes)",
          "method", 0),
    chunk("fastapi/routing.py", "APIRouter",
          "class APIRouter(routing.Router): a router groups path operations", "class", 1),
    chunk("fastapi/applications.py", "FastAPI.add_api_route",
          "def add_api_route(self, path, endpoint): self.router.add_api_route(path, endpoint)",
          "method", 0),
    chunk("docs_src/bigger_applications/tutorial001.py", "",
          "from fastapi import APIRouter\nrouter = APIRouter()", "module", 0),
    chunk("tests/test_routing.py", "test_include_router",
          "def test_include_router(): assert client.get('/').status_code == 200", "function", 0),
    chunk("fastapi/params.py", "Depends",
          "class Depends: dependency injection marker", "class", 0),
    chunk("static/style.css", "", "body { margin: 0 }", "block", 0, "css", ".css"),
]


def embedding(seed: float = 0.1) -> list[float]:
    return [seed] * DIM


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    store = VectorStore(persist_directory=str(tmp_path / "chroma"))
    store.add_chunks(REPO, CORPUS, [embedding(i / 20) for i in range(len(CORPUS))])
    return store


# =========================================================== item 5: filters


def test_build_where_returns_none_when_empty():
    assert build_where({}) is None
    assert build_where({"language": None}) is None


def test_build_where_single_clause():
    assert build_where({"language": "python"}) == {"language": {"$eq": "python"}}


def test_build_where_combines_with_and():
    where = build_where({"language": "python", "extension": ".py"})
    assert "$and" in where
    assert len(where["$and"]) == 2


def test_build_where_uses_in_for_lists():
    assert build_where({"chunk_type": ["function", "method"]}) == {
        "chunk_type": {"$in": ["function", "method"]}
    }


def test_build_where_collapses_single_element_list():
    assert build_where({"language": ["python"]}) == {"language": {"$eq": "python"}}


def test_build_where_rejects_unknown_field():
    with pytest.raises(ValueError, match="not filterable"):
        build_where({"nonsense": "x"})


def test_build_where_rejects_empty_list():
    with pytest.raises(ValueError, match="empty"):
        build_where({"language": []})


def test_repository_is_not_filterable():
    # Isolation comes from collection choice, not an omittable filter.
    assert "repository" not in FILTERABLE_FIELDS


def test_matches_mirrors_where_semantics():
    metadata = {"language": "python", "chunk_type": "method"}
    assert matches(metadata, {"language": "python"})
    assert matches(metadata, {"chunk_type": ["method", "function"]})
    assert not matches(metadata, {"language": "go"})
    assert matches(metadata, None)


def test_vector_retriever_applies_language_filter(store: VectorStore):
    results = VectorRetriever(store).retrieve(
        embedding(0.1), REPO, top_k=10, filters={"language": "css"}
    )
    assert results
    assert all(r["language"] == "css" for r in results)


def test_vector_retriever_applies_chunk_type_filter(store: VectorStore):
    results = VectorRetriever(store).retrieve(
        embedding(0.1), REPO, top_k=10, filters={"chunk_type": ["method", "class"]}
    )
    assert all(r["chunk_type"] in {"method", "class"} for r in results)


def test_retriever_facade_passes_filters_through(store: VectorStore):
    results = Retriever(store).retrieve(embedding(0.1), REPO, top_k=10, language="css")
    assert all(r["language"] == "css" for r in results)


# ======================================================== item 6: tokenising


def test_split_identifier_handles_camel_case():
    assert set(split_identifier("includeRouter")) >= {"include", "router", "includerouter"}


def test_split_identifier_handles_snake_case():
    assert set(split_identifier("include_router")) >= {"include", "router"}


def test_split_identifier_handles_acronyms():
    assert set(split_identifier("APIRouter")) >= {"api", "router"}


def test_tokenize_path_drops_the_extension():
    assert tokenize_path("fastapi/routing.py") == ["fastapi", "routing"]


def test_tokenize_removes_stopwords():
    assert "how" not in tokenize("how does routing work")
    assert "routing" in tokenize("how does routing work")


def test_chunk_terms_weight_path_and_symbol():
    terms = chunk_terms(CORPUS[0])
    # "routing" appears once in the path, repeated by the path weight.
    assert terms.count("routing") >= 3
    assert terms.count("include") >= 4


# ============================================================= item 6: BM25


def test_bm25_finds_the_file_named_after_the_query():
    index = BM25Index.build(CORPUS)
    best = index.search("routing", top_k=1)[0][0]
    assert best["relative_path"] == "fastapi/routing.py"


def test_bm25_matches_symbols():
    index = BM25Index.build(CORPUS)
    best = index.search("include_router", top_k=1)[0][0]
    assert best["symbol"] == "APIRouter.include_router"


def test_bm25_returns_nothing_for_absent_terms():
    assert BM25Index.build(CORPUS).search("kubernetes helm chart", top_k=5) == []


def test_bm25_scores_are_non_negative():
    index = BM25Index.build(CORPUS)
    assert all(s >= 0 for s in index.score_query("routing dependency"))


def test_bm25_handles_an_empty_corpus():
    index = BM25Index.build([])
    assert len(index) == 0
    assert index.search("anything") == []


def test_bm25_idf_stays_positive_for_common_terms():
    # Unsmoothed IDF goes negative past 50% document frequency, which would
    # let a common term penalise a document.
    corpus = [chunk(f"a{i}.py", "", "router router router", index=i) for i in range(10)]
    index = BM25Index.build(corpus)
    assert all(s >= 0 for s in index.score_query("router"))


def test_keyword_retriever_uses_the_stored_corpus(store: VectorStore):
    results = KeywordRetriever(store).retrieve("routing", REPO, top_k=3)
    assert results
    assert results[0]["relative_path"] == "fastapi/routing.py"


def test_keyword_retriever_respects_filters(store: VectorStore):
    results = KeywordRetriever(store).retrieve(
        "router", REPO, top_k=10, filters={"chunk_type": "module"}
    )
    assert all(r["chunk_type"] == "module" for r in results)


def test_keyword_index_is_cached_and_invalidated(store: VectorStore):
    retriever = KeywordRetriever(store)
    first = retriever.index_for(REPO)
    assert retriever.index_for(REPO) is first
    retriever.invalidate(REPO)
    assert retriever.index_for(REPO) is not first


# =========================================================== item 6: fusion


def test_normalise_maps_onto_unit_range():
    assert normalise([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]


def test_normalise_handles_identical_values():
    assert normalise([2.0, 2.0]) == [1.0, 1.0]
    assert normalise([0.0, 0.0]) == [0.0, 0.0]


def test_normalise_handles_empty():
    assert normalise([]) == []


def test_fusion_keeps_both_score_channels(store: VectorStore):
    hybrid = HybridRetriever(store)
    results = hybrid.retrieve("include_router", embedding(0.0), REPO, top_k=7)
    assert any(r["vector_score"] > 0 for r in results)
    assert any(r["keyword_score"] > 0 for r in results)


def test_fusion_deduplicates_chunks_found_by_both(store: VectorStore):
    results = HybridRetriever(store).retrieve("routing", embedding(0.0), REPO, top_k=10)
    keys = [(r["relative_path"], r["chunk_index"]) for r in results]
    assert len(keys) == len(set(keys))


def test_hybrid_beats_dense_alone_at_finding_the_named_file(store: VectorStore):
    query = "How does routing work?"
    dense = VectorRetriever(store).retrieve(embedding(0.9), REPO, top_k=3)
    hybrid = HybridRetriever(store).retrieve(query, embedding(0.9), REPO, top_k=3)

    dense_paths = [r["relative_path"] for r in dense]
    hybrid_paths = [r["relative_path"] for r in hybrid]
    # The keyword channel is what puts the file named after the query on top.
    assert "fastapi/routing.py" in hybrid_paths
    assert hybrid_paths != dense_paths or "fastapi/routing.py" in dense_paths


def test_hybrid_respects_top_k(store: VectorStore):
    assert len(HybridRetriever(store).retrieve("router", embedding(0.0), REPO, top_k=2)) == 2


def test_hybrid_applies_filters(store: VectorStore):
    results = HybridRetriever(store).retrieve(
        "router", embedding(0.0), REPO, top_k=10, language="python"
    )
    assert all(r["language"] == "python" for r in results)


# ======================================================== item 7: reranking


def test_reranker_prefers_implementation_over_tests():
    assert path_prior("fastapi/routing.py") == 1.0
    assert path_prior("tests/test_routing.py") < 1.0
    assert path_prior("docs_src/tutorial001.py") < 1.0


def test_reranker_demotes_module_chunks():
    reranker = HeuristicReranker()
    terms = set(tokenize("routing"))
    function_chunk = dict(CORPUS[0], vector_score=0.5, keyword_score=0.5)
    module_chunk = dict(CORPUS[3], vector_score=0.5, keyword_score=0.5)

    assert reranker.combine(reranker.features(terms, function_chunk)) > reranker.combine(
        reranker.features(terms, module_chunk)
    )


def test_reranker_rewards_symbol_overlap():
    reranker = HeuristicReranker()
    terms = set(tokenize("include router"))
    features = reranker.features(terms, dict(CORPUS[0], vector_score=0, keyword_score=0))
    assert features["symbol_overlap"] > 0


def test_reranker_rewards_path_overlap():
    reranker = HeuristicReranker()
    terms = set(tokenize("routing"))
    features = reranker.features(terms, dict(CORPUS[0], vector_score=0, keyword_score=0))
    assert features["path_overlap"] > 0


def test_reranker_attaches_an_explanation():
    results = HeuristicReranker().rerank(
        "routing", [dict(c, vector_score=0.5, keyword_score=0.2) for c in CORPUS], top_k=3
    )
    assert all("rerank_features" in r for r in results)
    assert all(r["score"] == r["rerank_score"] for r in results)


def test_reranker_orders_by_score():
    results = HeuristicReranker().rerank(
        "include_router", [dict(c, vector_score=0.4, keyword_score=0.3) for c in CORPUS], top_k=7
    )
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_reranker_handles_an_empty_pool():
    assert HeuristicReranker().rerank("x", [], top_k=5) == []


def test_identity_reranker_is_a_control():
    pool = [dict(c, vector_score=i / 10, keyword_score=0.0) for i, c in enumerate(CORPUS)]
    results = IdentityReranker().rerank("routing", pool, top_k=3)
    assert results[0]["vector_score"] >= results[-1]["vector_score"]


def test_reranker_is_swappable(store: VectorStore):
    hybrid = HybridRetriever(store, reranker=IdentityReranker())
    assert hybrid.retrieve("routing", embedding(0.0), REPO, top_k=3)
