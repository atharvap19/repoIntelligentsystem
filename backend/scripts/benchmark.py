"""Phase 2 retrieval benchmark.

Measures real behaviour against a real index. Every number this prints comes
from a live run — nothing is estimated, and where a comparison is not
available the script says so rather than inventing one.

What it measures, per query:

* whether the *implementation* file appears at all, and at what rank;
* retrieval latency (embedding + search + rerank), excluding generation;
* the composition of the result set, since the failure mode that motivated
  Phase 2 was tutorials and tests crowding out library source.

Usage::

    python scripts/benchmark.py --repository fastapi
    python scripts/benchmark.py --repository fastapi --compare repositories
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.reranker import FusionOnlyReranker  # noqa: E402
from app.rag.retriever import HybridRetriever, VectorRetriever  # noqa: E402
from app.rag.vector_store import VectorStore  # noqa: E402
from app.services.embedding_service import EmbeddingService  # noqa: E402

#: Each query names the file a correct answer must be grounded in. These are
#: judgements about FastAPI's layout, stated up front so the scoring is
#: auditable rather than hidden in the numbers.
QUERIES: list[dict] = [
    {
        "question": "How does FastAPI handle routing?",
        "expect_prefix": "fastapi/routing.py",
    },
    {
        "question": "How does APIRouter include another router?",
        "expect_prefix": "fastapi/routing.py",
    },
    {
        "question": "How are dependencies resolved?",
        "expect_prefix": "fastapi/dependencies/",
    },
    {
        "question": "How is a response serialized into JSON?",
        "expect_prefix": "fastapi/routing.py",
    },
    {
        "question": "How does the OpenAPI schema get generated?",
        "expect_prefix": "fastapi/openapi/",
    },
    {
        "question": "How is request body validation performed?",
        "expect_prefix": "fastapi/",
    },
    {
        "question": "How does security and authentication work?",
        "expect_prefix": "fastapi/security/",
    },
    {
        "question": "How are background tasks executed?",
        "expect_prefix": "fastapi/background.py",
    },
    {
        "question": "How are websockets supported?",
        "expect_prefix": "fastapi/",
    },
    {
        "question": "How does exception handling work?",
        "expect_prefix": "fastapi/exception_handlers.py",
    },
]

TOP_K = 5
DEEP_K = 50


def normalise_path(path: str) -> str:
    """Compare paths separator-agnostically.

    Pre-Phase-2 collections stored Windows separators (``fastapi\\routing.py``).
    Matching them against forward-slash prefixes silently scores every query as
    a miss, which reads as a catastrophic result rather than the measurement
    bug it is.
    """
    return str(path or "").replace("\\", "/")


def rank_of(results: list[dict], prefix: str) -> int | None:
    """1-based rank of the first result under ``prefix``, or ``None``."""
    for position, result in enumerate(results, start=1):
        if normalise_path(result.get("relative_path", "")).startswith(prefix):
            return position
    return None


def area_of(path: str) -> str:
    return normalise_path(path).split("/")[0] or "?"


def evaluate(name: str, retrieve, queries: list[dict]) -> dict:
    """Run every query through one strategy and summarise."""
    rows: list[dict] = []
    latencies: list[float] = []

    for spec in queries:
        started = time.perf_counter()
        deep = retrieve(spec["question"], DEEP_K)
        elapsed = (time.perf_counter() - started) * 1000
        latencies.append(elapsed)

        top = deep[:TOP_K]
        rows.append(
            {
                "question": spec["question"],
                "expect": spec["expect_prefix"],
                "rank_in_top_k": rank_of(top, spec["expect_prefix"]),
                "rank_in_deep": rank_of(deep, spec["expect_prefix"]),
                "latency_ms": round(elapsed, 1),
                "areas": Counter(area_of(r["relative_path"]) for r in top),
                "types": Counter(str(r.get("chunk_type", "")) for r in top),
            }
        )

    hits_at_k = sum(1 for r in rows if r["rank_in_top_k"] is not None)
    hits_at_deep = sum(1 for r in rows if r["rank_in_deep"] is not None)
    found_ranks = [r["rank_in_deep"] for r in rows if r["rank_in_deep"]]

    # Mean reciprocal rank over the deep list; a miss contributes 0.
    mrr = statistics.mean(
        [1.0 / r["rank_in_deep"] if r["rank_in_deep"] else 0.0 for r in rows]
    )

    return {
        "strategy": name,
        "queries": len(rows),
        f"hit@{TOP_K}": hits_at_k,
        f"hit@{DEEP_K}": hits_at_deep,
        "mrr": round(mrr, 3),
        "median_rank_when_found": statistics.median(found_ranks) if found_ranks else None,
        "median_latency_ms": round(statistics.median(latencies), 1),
        "rows": rows,
    }


def print_report(report: dict) -> None:
    print(f"\n=== {report['strategy']} ===")
    print(f"  queries              : {report['queries']}")
    print(f"  hit@{TOP_K}               : {report[f'hit@{TOP_K}']}/{report['queries']}")
    print(f"  hit@{DEEP_K}              : {report[f'hit@{DEEP_K}']}/{report['queries']}")
    print(f"  MRR                  : {report['mrr']}")
    print(f"  median rank (found)  : {report['median_rank_when_found']}")
    print(f"  median latency       : {report['median_latency_ms']}ms")
    print("  per query:")
    for row in report["rows"]:
        top = row["rank_in_top_k"]
        deep = row["rank_in_deep"]
        mark = "HIT " if top else ("deep" if deep else "MISS")
        rank = f"#{deep}" if deep else "-"
        print(f"    [{mark}] {rank:>4}  {row['question'][:52]:<52} {dict(row['types'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="fastapi")
    parser.add_argument(
        "--compare",
        default=None,
        help="A second collection to run dense-only against, e.g. a pre-Phase-2 index.",
    )
    parser.add_argument("--json", default=None, help="Write the full report here.")
    args = parser.parse_args()

    store = VectorStore()
    available = store.list_repositories()
    if args.repository not in available:
        print(f"Repository {args.repository!r} not indexed. Available: {available}")
        return 1

    embedder = EmbeddingService()
    cache: dict[str, list[float]] = {}

    def embed(question: str) -> list[float]:
        if question not in cache:
            cache[question] = embedder.generate_embedding(question)
        return cache[question]

    # Warm every embedding before timing anything. Otherwise whichever
    # strategy runs first absorbs the entire embedding cost and looks an
    # order of magnitude slower than the others.
    print("Warming query embeddings...")
    for spec in QUERIES:
        embed(spec["question"])

    dense = VectorRetriever(store)
    hybrid = HybridRetriever(store)
    control = HybridRetriever(store, reranker=FusionOnlyReranker())

    reports = [
        evaluate(
            f"dense only  [{args.repository}]",
            lambda q, k: dense.retrieve(embed(q), args.repository, top_k=k),
            QUERIES,
        ),
        evaluate(
            f"hybrid fusion only  [{args.repository}]",
            lambda q, k: control.retrieve(q, embed(q), args.repository, top_k=k),
            QUERIES,
        ),
        evaluate(
            f"hybrid + rerank  [{args.repository}]",
            lambda q, k: hybrid.retrieve(q, embed(q), args.repository, top_k=k),
            QUERIES,
        ),
    ]

    if args.compare and args.compare in available:
        legacy = VectorRetriever(store)
        reports.insert(
            0,
            evaluate(
                f"dense only  [{args.compare}]  (pre-Phase-2 index)",
                lambda q, k: legacy.retrieve(embed(q), args.compare, top_k=k),
                QUERIES,
            ),
        )
    elif args.compare:
        print(f"\nNote: comparison collection {args.compare!r} is not present; skipping it.")

    for report in reports:
        print_report(report)

    print("\n=== summary ===")
    print(f"  {'strategy':<52} {'hit@5':>6} {'hit@50':>7} {'MRR':>6} {'p50 ms':>8}")
    for report in reports:
        print(
            f"  {report['strategy']:<52} "
            f"{report[f'hit@{TOP_K}']:>6} {report[f'hit@{DEEP_K}']:>7} "
            f"{report['mrr']:>6} {report['median_latency_ms']:>8}"
        )

    if args.json:
        serialisable = [
            {**r, "rows": [{**row, "areas": dict(row["areas"]), "types": dict(row["types"])}
                           for row in r["rows"]]}
            for r in reports
        ]
        Path(args.json).write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
