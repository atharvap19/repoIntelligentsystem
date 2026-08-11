# LangChain integration — component-by-component evaluation

The brief was explicit: *evaluate each component instead of blindly replacing
everything*, keep repository-specific logic custom, and do not end up with two
permanent RAG implementations. This records what was adopted, what was not,
and why.

The rule applied throughout: **adopt LangChain where the problem is generic;
keep custom where the problem is about repositories.**

## Adopted

| Component | Used for | Why |
| --- | --- | --- |
| `ChatOllama` | Answer generation | Message construction, streaming, retries and timeouts are generic client concerns. The previous hand-rolled `ollama.Client` call did none of them. |
| `ChatPromptTemplate` | System + user prompt | Turns prompt assembly into a declared, testable structure with named slots instead of an f-string built inside the service. |
| `StrOutputParser` | Response extraction | Removes `response["message"]["content"]` indexing, which was a shape assumption about Ollama's payload. |
| LCEL (`prompt \| llm \| parser`) | Generation pipeline | Gives streaming for free (`.stream()`) alongside `.invoke()` with no second code path. |

That is the whole of it, and it is confined to
[`llm_service.py`](../backend/app/services/llm_service.py).

## Not adopted, with reasons

**`Chroma` vector store wrapper.** The wrapper models one collection per store
instance and owns ID generation. This project needs one collection *per
repository* resolved at query time, deterministic repository-scoped IDs, and
`upsert` for idempotent re-indexing. Adopting it would mean fighting it on all
three, and the custom store is ~200 lines that do exactly what is needed.

**`OllamaEmbeddings`.** The existing `EmbeddingService` already batches through
Ollama's native `embed` endpoint. The wrapper adds an abstraction layer without
changing the model, the endpoint, or the vectors. No benefit to weigh against
the churn of re-indexing.

**`BaseRetriever`.** LangChain retrievers return `Document` objects with a
`page_content`/`metadata` split. Our results carry `vector_score`,
`keyword_score`, `rerank_score` and a `rerank_features` breakdown — the data
that makes a ranking decision explainable. Flattening that into
`Document.metadata` to satisfy an interface nothing currently consumes would
lose the shape for no gain. Worth revisiting in Phase 3 if LangGraph nodes need
to consume a retriever directly.

**Text splitters.** `RecursiveCharacterTextSplitter` is a better character
splitter than ours, but this project does not want a character splitter. AST
chunking produces `chunk_type`, `symbol`, `start_line` and `end_line`, which is
the metadata the reranker and the prompt both depend on.

**Document loaders.** Source filtering is the product. `DirectoryLoader` has no
concept of "skip vendored bundles, skip generated protobufs, skip prose,
prefer implementation", and that filtering removed 64% of FastAPI's files.

## Where the chain stops

`ChatService` orchestrates; the chain generates. Repository resolution, hybrid
retrieval, filtering and response shaping stay in application code, so the LCEL
pipeline is `text in → text out`. This is deliberate — the brief asked not to
put the entire application inside one chain, and keeping the boundary there is
what lets retrieval be benchmarked without invoking an LLM at all.

## Cost

Three packages: `langchain-core`, `langchain-ollama`, and their shared
dependencies. No paid APIs, nothing leaves the machine, and Ollama remains the
only model runtime.
