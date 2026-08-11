/** Shared contracts between the UI and the FastAPI backend. */

/** One retrieved chunk.
 *
 *  `file` is the primary path key and is always present. The rest became
 *  available once Phase 2 persisted AST metadata and Phase 3 split the
 *  retrieval score into its channels, so they stay optional for the demo
 *  fixtures and for any older backend.
 */
export interface Source {
  file: string
  distance: number
  content?: string
  language?: string
  chunk_index?: number
  repository?: string
  /** Same value as `file`; returned alongside it for symmetry with the graph. */
  relative_path?: string
  file_name?: string
  /** Qualified symbol, e.g. `APIRouter.include_router`. */
  symbol?: string
  chunk_type?: string
  start_line?: number
  end_line?: number
  /** Reranked score actually used for ordering. */
  score?: number
  /** The two fused channels, for explaining why a chunk ranked where it did. */
  vector_score?: number
  keyword_score?: number
}

export interface ChatResponse {
  question: string
  answer: string
  sources: Source[]
  /** Optional per-stage timings, in ms, once the backend reports them. */
  timings?: Partial<Record<PipelineStage, number>>
}

export type PipelineStage = 'embed' | 'search' | 'generate'

export interface ImportResponse {
  status: 'success' | 'error'
  message: string
  repository?: string
  path?: string
}

export interface IndexStats {
  files: number
  chunks: number
  stored_vectors: number
}

export type IndexState = 'unindexed' | 'cloning' | 'indexing' | 'ready' | 'error'

export interface Repository {
  id: string
  name: string
  url?: string
  path?: string
  state: IndexState
  stats?: IndexStats
  /** Percentage of source files per language, for the composition bar. */
  languages?: Record<string, number>
  error?: string
  importedAt: number
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  timings?: Partial<Record<PipelineStage, number>>
  /** Wall-clock round trip measured in the browser. */
  latencyMs?: number
  error?: string
  pending?: boolean
  repositoryId?: string
  createdAt: number
}

export interface RunSettings {
  topK: number
  model: string
  embedModel: string
  /** Hide chunks whose relevance falls under this bar, in the Inspector only. */
  minRelevance: number
}

export type ConnectionState = 'checking' | 'online' | 'offline' | 'demo'
