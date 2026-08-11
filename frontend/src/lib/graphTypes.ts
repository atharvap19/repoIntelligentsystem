/** Contracts for the Phase 3 graph, timeline and agent endpoints. */

export type NodeKind =
  | 'repository'
  | 'module'
  | 'directory'
  | 'file'
  | 'class'
  | 'function'
  | 'method'
  | 'external'

export type EdgeKind =
  | 'CONTAINS'
  | 'IMPORTS'
  | 'DEPENDS_ON'
  | 'CALLS'
  | 'EXTENDS'
  | 'IMPLEMENTS'

/** Free-form per-node payload. Which keys exist depends on `kind`. */
export interface GraphNodeData {
  language?: string
  extension?: string
  module?: string
  directory?: string
  size?: number
  line_count?: number
  class_count?: number
  function_count?: number
  import_count?: number
  complexity?: 'low' | 'medium' | 'high'
  parse_failed?: boolean
  /** Symbols only. */
  relative_path?: string
  start_line?: number
  end_line?: number
  is_async?: boolean
  bases?: string[]
  /** Modules only — rolled up from the subtree. */
  file_count?: number
  languages?: Record<string, number>
  primary_language?: string
  first_seen?: number | null
  /** Snapshot only: files present at the cursor. */
  file_count_at?: number
  /** Externals only. */
  uses?: number
}

export interface GraphNode {
  id: string
  repository: string
  kind: NodeKind
  name: string
  key: string
  parent_id: string | null
  level: number
  data: GraphNodeData
}

export interface GraphEdge {
  id: string
  repository: string
  kind: EdgeKind
  source: string
  target: string
  data: Record<string, unknown>
}

export interface GraphOverview {
  repository: string
  root: GraphNode
  nodes: GraphNode[]
  edges: GraphEdge[]
  counts: Record<string, number>
  timeline: { first_at: number | null; last_at: number | null; total: number }
}

export interface GraphExpansion {
  node: GraphNode
  nodes: GraphNode[]
  edges: GraphEdge[]
  total_children: number
  truncated: boolean
}

export interface CommitRef {
  sha: string
  author: string
  authored_at: number
  summary: string
  change_type?: string
}

/** A node returned as the endpoint of a dependency edge. */
export interface RelatedNode extends GraphNode {
  via: EdgeKind
  edge_data: Record<string, unknown>
}

export interface NodeDetail {
  node: GraphNode
  symbols?: GraphNode[]
  dependencies?: RelatedNode[]
  dependents?: RelatedNode[]
  history?: CommitRef[]
  children?: GraphNode[]
  parent?: GraphNode | null
  children_count?: number
}

export interface DependencyResult {
  node: GraphNode
  dependencies: RelatedNode[]
  dependents: RelatedNode[]
  edges: GraphEdge[]
}

export interface FileContent {
  node: GraphNode
  relative_path: string
  language: string
  content: string
  line_count: number
  truncated: boolean
  redacted: boolean
  redaction_findings: string[]
  symbols: GraphNode[]
}

export interface TimelinePeriod {
  period: string
  commits: number
  changes: number
  first_at: number
  last_at: number
}

export interface TimelineLane {
  module: string
  first_seen: number | null
  periods: TimelinePeriod[]
  total_commits: number
}

export interface Timeline {
  repository: string
  range: { first_at: number | null; last_at: number | null; total: number }
  lanes: TimelineLane[]
  releases: { name: string; sha: string; created_at: number }[]
  branches: { name: string; sha: string; created_at: number }[]
}

export interface Snapshot {
  repository: string
  timestamp: number
  root: GraphNode | null
  nodes: GraphNode[]
  edges: GraphEdge[]
  files_present: number
  approximate: boolean
}

/** Instruction from the agent for the UI to act on. */
export type UiAction =
  | { type: 'open_file'; node_id: string }
  | { type: 'show_structure'; repository: string }
  | {
      type: 'highlight_dependencies'
      node_id: string
      dependencies: string[]
      dependents: string[]
    }

export interface AgentResponse {
  question: string
  repository: string
  conversation_id: string
  intent: string
  confidence: number
  answer: string
  focus: { node_id: string | null; label: string | null }
  ui_action: UiAction | null
  graph: unknown
  sources: import('./types').Source[]
  trace: { node: string; [key: string]: unknown }[]
}
