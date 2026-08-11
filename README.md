# Repository Intelligence Platform

**An AI copilot for understanding unfamiliar codebases.**

Point it at a GitHub repository. It reads the source, builds a searchable index
and a structural graph, and then answers questions about the code — showing you
exactly which lines produced every answer.

Everything runs locally. Ollama serves the models, ChromaDB stores the vectors,
SQLite stores the graph. No paid APIs, no code leaves the machine.

---

## Contents

- [What problem this solves](#what-problem-this-solves)
- [Running it](#running-it)
- [Architecture](#architecture)
- [Phase 1 — the working baseline](#phase-1--the-working-baseline)
- [Phase 2 — production retrieval](#phase-2--production-retrieval)
- [Phase 3 — the repository graph](#phase-3--the-repository-graph)
- [Measured results](#measured-results)
- [Engineering decisions and trade-offs](#engineering-decisions-and-trade-offs)
- [Testing](#testing)
- [Known limitations](#known-limitations)
- [Phase 4](#phase-4)

---

## What problem this solves

A developer joining an unfamiliar repository asks two kinds of question:

1. **Understanding** — *"How does routing work?"*, *"What is this repo about?"*
2. **Implementation** — *"Where do I add an endpoint?"*, *"What depends on this class?"*

Ordinary code search answers neither well. Grep finds strings, not concepts. A
plain RAG chatbot finds text that *looks* similar to the question and guesses.

This system does three things instead:

| | |
| --- | --- |
| **Hybrid retrieval** | Semantic similarity fused with keyword matching over file paths and symbol names |
| **Structural graph** | Every file, class and function as a node; every `import` as an edge — so dependency questions are *lookups*, not searches |
| **Intent routing** | The question type decides the strategy; structural questions read the graph, code questions read retrieved chunks |

And every answer shows its evidence: the exact chunks, their files, their line
ranges, and their retrieval scores.

---

## Running it

**Prerequisites:** Python 3.11+, Node 20+, and [Ollama](https://ollama.com).

```bash
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

**Backend**

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend**

```bash
cd frontend
npm install
npm run dev            # http://localhost:5173
```

**Index a repository**

```python
from app.services.indexing_service import IndexingService

service = IndexingService(batch_size=64)
service.index_repository("repositories/fastapi")    # full build
service.update_repository("repositories/fastapi")   # only what changed
service.reindex_repository("fastapi")               # drop and rebuild
```

Or over HTTP: `POST /github/import` then `POST /index/`.

The UI has two tabs. **Chat** is question-and-answer with a retrieval inspector.
**Explorer** is the visual graph, code viewer and timeline.

---

## Architecture

```
GitHub URL
    │
    ▼
GitHubService ──────────────► repositories/<name>/          (git clone)
    │
    ▼
RepositoryParser ───────────► 15 source extensions only
    │                          prunes .git, node_modules, dist, vendor…
    │                          drops prose, binaries, lockfiles, generated files
    ▼
RepositoryChunker ──────────► Python split along its syntax tree (ast)
    │                          other languages fall back to fixed-size
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
EmbeddingService                    GraphBuilder + GitHistory
 (nomic-embed-text, batched)         (symbols, imports, commits)
    ▼                                      ▼
ChromaDB                             SQLite
 one collection per repository        nodes / edges / commits
    │                                      │
    └──────────────┬───────────────────────┘
                   ▼
            LangGraph agent
                   │
            ┌──────┴──────┐
            ▼             ▼
   HybridRetriever   Graph queries
   (vector + BM25     (structure, dependencies,
    + reranking)       history)
            └──────┬──────┘
                   ▼
        prompt │ ChatOllama │ StrOutputParser        (LangChain LCEL)
                   ▼
          answer + evidence + UI actions
```

**Stack**

| Layer | Choice |
| --- | --- |
| API | FastAPI |
| Vectors | ChromaDB (one collection per repository) |
| Graph | SQLite |
| Embeddings | `nomic-embed-text` via Ollama |
| Generation | `qwen3:4b` via Ollama, behind a provider abstraction |
| LLM plumbing | LangChain LCEL |
| Orchestration | LangGraph |
| Frontend | React 18, Vite, TypeScript, Tailwind, Zustand, React Flow |

**Size:** ~6,800 lines backend · ~2,900 lines tests · ~4,700 lines frontend.

---

## Phase 1 — the working baseline

A straight-line RAG pipeline: clone → parse → chunk (1000 characters, 100
overlap) → embed → store in one Chroma collection → retrieve top-5 → answer.

It worked, and that mattered — it gave us something real to measure. Measuring
it is what produced Phase 2.

**The failure that defined the next phase.** Asked *"How does FastAPI handle
routing?"*, it returned Portuguese and Chinese documentation translations. The
file that actually implements routing never appeared.

Diagnosis: **markdown was 64% of the indexed corpus** — 1,691 of 2,647 files.
Prose outnumbered code 1.8 : 1, so prose won every ranking.

---

## Phase 2 — production retrieval

Twelve items, each measured before and after.

### Source filtering

A whitelist of 15 source extensions. Directory pruning happens *during* the
walk, so `.git` and `node_modules` are never descended into. Files are excluded
by name (`README`, `LICENSE`, `mkdocs.yml`), by stem, by binary extension, by
generated-file suffix (`.min.js`, `_pb2.py`), by size, and by NUL-byte content
sniffing. Every exclusion is counted and attributable.

**Result: 2,647 → 956 files.** Parse time 1.00s → 0.36s.

### AST chunking

Python files are split along their syntax tree instead of every 1000
characters. A chunk is now a function, a method, a class, or a run of
module-level code — carrying `chunk_type`, `symbol`, `start_line`, `end_line`.

Decorators are pulled into their function's span, because `@app.get("/users")`
is exactly the text that makes a route handler findable. Comment blocks above a
definition are absorbed rather than dropped between chunks.

**Result: 82.5% of chunks now carry a symbol name.** `APIRouter.include_router`
is addressable as itself.

### Per-repository collections

Phase 1 used one collection and IDs like `README.md_0` — no repository prefix,
so two repositories both containing `src/main.py` silently overwrote each other.
Now each repository gets its own Chroma collection, and isolation is
*structural*: a query against `fastapi` cannot return a `httpx` chunk because
that collection is never opened.

Writes use `upsert`, so re-indexing is idempotent instead of an ID collision.

### Hybrid retrieval — the biggest win

Dense embeddings are blind to exact names. The token "routing" *is* the answer
to "how does routing work", but cosine similarity has no way to know that
`fastapi/routing.py` is the file named after the question.

So: **Okapi BM25 implemented directly** (~40 lines, no dependency) over a
code-aware tokenizer that splits `camelCase` and `snake_case`, and weights path
and symbol tokens above body text. Candidates from both channels are fused,
then reranked.

### Reranking

The hardware is a 4 GB laptop GPU already hosting two models, so a cross-encoder
would contend for VRAM on every query. Instead: a **feature-based reranker** —
dense score, keyword score, symbol overlap, path overlap, plus priors that
demote `module` chunks and test/example paths. No model, no VRAM,
sub-millisecond, and behind a `Reranker` protocol so a cross-encoder can replace
it later.

### Incremental indexing

SHA-256 content hashes per file, stored in chunk metadata rather than a side
manifest so they cannot drift from the vectors they describe. Unchanged files
cost zero embeddings.

Stale chunks are deleted **before** the rewrite — a file that shrinks would
otherwise leave orphaned high-index chunks that `upsert` never touches. There's
a test for exactly that.

### LangChain, selectively

Adopted where the problem is generic: `ChatOllama`, `ChatPromptTemplate`,
`StrOutputParser`, LCEL.

Rejected with reasons: the Chroma wrapper (fights per-repository collections),
`OllamaEmbeddings` (no benefit over existing batching), `BaseRetriever` (would
flatten the score breakdown), text splitters and document loaders (the custom
versions *are* the product). Recorded in [`docs/langchain.md`](docs/langchain.md).

---

## Phase 3 — the repository graph

Phase 2 made retrieval good. It could not make it *complete*, because some
questions have no answer in any chunk.

Asked *"What is this repository about?"*, the Phase 2 system answered:

> "This repository is a FastAPI application that handles GitHub repository
> translations and notifications…"

Wrong. It had retrieved FastAPI's internal maintainer scripts, because the word
"repository" in the question matched class names like `CommentsRepository`.

**No chunk *is* a repository's architecture.** You cannot retrieve what was
never written. So Phase 3 computes it.

### The graph

Every file, class, function and method becomes a **node**. Every `import`
becomes an **edge**. Built by walking the same AST the chunker already parses,
so the graph and the vector index always describe the same corpus.

```
Repository ── CONTAINS ──► Module ── CONTAINS ──► Directory
                                                      │
                                                  CONTAINS
                                                      ▼
                                                    File ── IMPORTS ──► File
                                                      │
                                                  CONTAINS
                                                      ▼
                                              Class / Function
                                                      │
                                                  CONTAINS
                                                      ▼
                                                   Method
```

8 node kinds, 10 edge kinds. **FastAPI: 6,363 nodes, 8,046 edges, 5,149 symbols
— built in 1.16 seconds.**

This turns dependency questions into lookups. *"What depends on routing.py?"*
returns **exactly 15 files**. Not "probably these" — fifteen, by following the
arrows.

### Git timeline

Commits, file changes, branches, tags and merges, normalised into SQLite.
Timeline lanes are **modules, not git branches**: a clone has one local branch
and the branch graph says little about how a codebase grew, whereas module
activity says a great deal.

### LangGraph intent routing

```
question ──► resolve context ──► classify ──┬──► search           ─┐
                                            ├──► explain          ─┤
                                            ├──► architecture     ─┤
                                            ├──► structure        ─┼──► answer
                                            ├──► dependencies     ─┤
                                            ├──► file             ─┤
                                            ├──► history          ─┤
                                            └──► generate code    ─┘
```

Classification is **heuristic, not an LLM call**. Generation costs ~150 seconds
on this hardware; spending a second call to *label* the question would double
every turn's latency to decide something a scored keyword match gets right.
There's a test asserting exactly one LLM call per turn, whatever the intent.

Conversational state is checkpointed per conversation, so *"what does **it**
depend on?"* resolves against the previous turn's focus.

### The visual explorer

React Flow canvas with hierarchical drill-down — repository → module →
directory → file → symbols. Only one level renders at a time, because 6,363
nodes on screen is unusable and slow.

Clicking a file opens its source in a **code cell below the graph** — the graph
never disappears. Dependency tracing highlights what a node imports and what
imports it, dimming everything else. A timeline cursor rewinds the repository:
drag it back and modules disappear as you pass before their first commit.

The AI panel's answers drive the canvas: ask *"what depends on this?"* and the
graph highlights while the model explains.

---

## Measured results

All figures are from live runs against the FastAPI repository. Reproduce with
`python scripts/benchmark.py --repository fastapi`.

### Indexing

| | Phase 1 | Phase 2/3 |
| --- | ---: | ---: |
| Files indexed | 2,647 | **956** |
| Chunks | ~16,868 | **7,235** |
| Chunks with a symbol name | 0 | **5,972 (82.5%)** |
| Parse time (warm) | 1.00s | **0.36s** |
| Graph nodes | — | **6,363** |
| Graph build | — | **1.16s** |
| Git history (1,500 commits) | — | **0.47s** |

### Retrieval quality

Ten questions, each scored on whether the *implementation* file is retrieved
and at what rank:

| Strategy | hit@5 | hit@50 | MRR | p50 latency |
| --- | ---: | ---: | ---: | ---: |
| Dense only (Phase 1 index) | 3/10 | 10/10 | 0.146 | 38.6ms |
| Dense only (Phase 2 index) | 4/10 | 8/10 | 0.263 | 38.3ms |
| Hybrid fusion | 8/10 | 9/10 | 0.621 | 28.9ms |
| **Hybrid + rerank** | **8/10** | **9/10** | **0.675** | **34.9ms** |

**MRR improved 4.6×.** Two honest observations:

- **Most of the gain is BM25, not the reranker.** Fusion alone moves MRR from
  0.263 to 0.621; the reranker adds 0.054. Worth knowing before investing in a
  heavier reranking model.
- **Finer chunks made dense-only *worse* on deep recall** (10/10 → 8/10). A
  broad question matches no single AST unit well. The keyword channel is what
  repairs it.

*Caveat: ten queries is a small set, the expected-file judgements are ours, and
the Phase 1 row differs in both corpus and chunking, so it is not a controlled
ablation.*

### The headline question

> **"What is this repository about?"**
>
> **Phase 1:** Portuguese and Chinese documentation translations.
> **Phase 2:** *"…handles GitHub repository translations and notifications."* — wrong.
> **Phase 3:** *"This repository is the source code for FastAPI, a modern Python web framework for building APIs… built on top of Starlette… uses Pydantic for data validation."*

---

## Engineering decisions and trade-offs

**SQLite for the graph, not Neo4j.** ChromaDB already ships SQLite, so this
added zero dependencies. Adjacency queries are indexed lookups on one table,
which is all this needs. *Cost:* deep transitive traversal will need recursive
CTEs. Behind a `GraphStore` class so a real graph database can replace it.

**Heuristic intent routing, not an LLM classifier.** Every LLM call costs ~150
seconds. The classifier reports its confidence, so an LLM fallback can be added
for the low-confidence tail without changing the interface.

**Feature-based reranker, not a cross-encoder.** 4 GB of VRAM already hosts
qwen3 and nomic-embed-text.

**Delete-then-insert for graph rebuilds, upsert for vectors.** A graph rebuild
must not leave nodes for deleted files, and reconciling that incrementally is
more error-prone than rebuilding something that takes 1.16 seconds.

**Markdown excluded by default.** It was the single largest source of retrieval
noise. Re-enable per repository with
`ParserConfig(supported_extensions=DEFAULT | {".md"})`.

**Secret redaction on every byte leaving the backend.** Imported repositories
are arbitrary third-party code. An earlier version matched unquoted values and
rewrote FastAPI's `token = ctx_var.set(x)` into `token = [REDACTED])`,
destroying valid code — it now requires quoted literals, because committed
secrets are string literals and an unquoted right-hand side is an expression.

---

## Testing

```bash
cd backend
pip install -r requirements-dev.txt
pytest
```

**337 tests across 10 files.** Unit tests never touch the network or the real
databases — every store test uses a temporary directory, and embeddings and LLM
calls are faked wherever the subject under test is orchestration rather than
model quality.

| Area | Coverage |
| --- | --- |
| Parser | filtering, pruning, hashing, metadata, error paths |
| Chunker | AST units, decorators, line numbers, fallbacks, sizing |
| Vector store | IDs, metadata, isolation, upsert, collections |
| Retrieval | filters, BM25, fusion, reranking |
| Indexing | lifecycle, incremental diffing, batching |
| Graph | model, analysis, building, persistence, history |
| Graph API | overview, expansion, dependencies, content, timeline |
| Agent | all 8 intents, conversational state, single-LLM-call budget |

Plus a **25-check Definition-of-Done script** exercising the full workflow
end-to-end against the live backend — all passing, including one real LLM answer.

### Bugs found by measurement, not by luck

A representative sample, all fixed and regression-tested:

- **`git log` with a literal NUL separator** → `ValueError: embedded null byte`.
  Switched to git's `%x00` placeholder: **31s → 0.25s, a 124× speedup**,
  verified byte-identical to the previous implementation.
- **Redaction destroying valid code** (above).
- **Test runs leaking fixture repositories into the real graph database** —
  seven of them. Fixtures now inject an explicit store.
- **`**counts` clobbering a repository's name** with an integer, because the
  counts dictionary had its own `repository` key.
- **Time-travel snapshots mixing denominators** — 1,555 historical paths shown
  against 8 indexed files.
- **Architecture output claiming FastAPI began in 2025** — the earliest commit
  in the *walked window*, not the project's start. Now explicitly labelled so
  the model cannot repeat it as fact.

---

## Known limitations

Stated plainly, because they matter for what comes next.

**Latency.** A full answer takes 2–6 minutes. Retrieval takes 35 milliseconds;
generation is everything else. `qwen3:4b` emits a reasoning trace before
answering, and runs mostly on CPU — the model is 3.5 GB loaded against 4 GB of
VRAM, so roughly a third of its layers sit on the CPU.

**Time travel is approximate.** A file counts as present from its first observed
commit onward; deletions are not modelled. Exact reconstruction means reading a
tree per commit.

**History is windowed.** 1,500 commits by default, so dates are window-relative,
not creation dates.

**Non-Python symbol extraction is regex-based.** Good enough to place a file in
the graph and draw its imports; not a real parser. `strategy_for(language)` is
the extension point for Tree-sitter.

**Distance metric.** Collections use Chroma's default L2 space, while
`nomic-embed-text` is trained for cosine. The parameter is wired
(`VectorStore(distance_space="cosine")`) but switching requires rebuilding every
collection, so it has not been made default without a measured comparison.

**Clone validation.** `clone_repository` does not validate the URL — scheme,
host and derived directory name all come from user input. The frontend
validates; the backend does not.

**The UI has not been visually reviewed.** Verified at API, compile and test
level only.

---

## Phase 4

In rough priority order:

1. **Latency** — disable qwen3's reasoning trace, or move to a non-reasoning
   model. The single biggest usability win available.
2. **Cosine distance** — rebuild and measure against the existing benchmark.
3. **Tree-sitter** — real symbol extraction for JavaScript, TypeScript, Go, Rust.
4. **Exact historical reconstruction** — per-commit trees for true time travel.
5. **Durable conversations** — currently in-process memory.
6. **Backend URL validation** — close the clone injection gap.
7. **Index-time repository summary** — one LLM call per index, so "what is this
   about" is a lookup rather than an inference.

---

## Repository layout

```
backend/
  app/
    api/          chat · github · graph · agent · indexing
    rag/          parser · chunker · vector_store · retriever · bm25 · reranker
    graph/        model · analyzer · builder · store · git_history
    agent/        LangGraph routing · intent classification
    tools/        search · file · architecture
    services/     indexing · chat · embedding · llm · graph · llm_provider
    prompts/      system prompt
    utils/        secret redaction
  tests/          337 tests
  scripts/        retrieval benchmark
frontend/
  src/
    components/   chat · inspector · graph · repos · layout
    lib/          API clients, types, formatting
    store/        chat state · graph state
docs/
  langchain.md    component-by-component adoption rationale
```
