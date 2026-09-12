/** Shared contracts between the UI and the FastAPI backend. */

export type NodeKind = 'file' | 'class' | 'function' | 'method'

export type EdgeKind = 'CONTAINS' | 'IMPORTS' | 'DEPENDS_ON' | 'CALLS' | 'INHERITS' | 'IMPLEMENTS'

export interface GraphNode {
  id: string
  kind: NodeKind
  name: string
  /** Path for files; `path::Qualified.name` for symbols. */
  key: string
  parent_id: string | null
  data: {
    language?: string
    line_count?: number
    /** Symbols only. */
    relative_path?: string
    start_line?: number
    end_line?: number
  }
}

export interface GraphEdge {
  kind: EdgeKind
  source: string
  target: string
}

/** `GET /graph/{repository}/knowledge` */
export interface KnowledgeGraph {
  repository: string
  nodes: GraphNode[]
  edges: GraphEdge[]
  counts: Record<string, number>
  /** Files and symbols in the repository, before the node cap. */
  total_nodes: number
  truncated: boolean
}

export type ImportState = 'cloning' | 'building_graph' | 'indexing_code' | 'ready' | 'error'

export interface RepositoryStatus {
  repository: string
  url: string
  state: ImportState
  message: string
  /** The knowledge graph exists and can be drawn and asked about. */
  graph_ready: boolean
  /** Code is embedded, so answers can quote source. */
  search_ready: boolean
  progress_done: number
  progress_total: number
  started_at: number
  finished_at: number | null
}

export interface RepositoryEntry {
  repository: string
  counts: Record<string, number>
  status: RepositoryStatus
}

/** One retrieved code chunk an answer was grounded in. */
export interface Source {
  file: string
  relative_path?: string
  file_name?: string
  repository?: string
  symbol?: string
  start_line?: number
  end_line?: number
}

/** The graph nodes an answer drew on, most central first. */
export interface AnswerHighlight {
  node_ids: string[]
  focus_node_id: string | null
}

/** What the user has selected in the graph, sent with every question. */
export interface SelectionContext {
  repository?: string
  file?: string
  symbol?: string
  node_id?: string
}

export interface AgentResponse {
  repository: string
  intent: string
  confidence: number
  answer: string
  focus: { node_id: string | null; label: string | null }
  highlight: AnswerHighlight
  sources: Source[]
  context_stats: Record<string, number>
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  pending?: boolean
  error?: string
  intent?: string
  focusLabel?: string | null
  highlight?: AnswerHighlight
  sources?: Source[]
}

export type ConnectionState = 'checking' | 'online' | 'offline'
