import { create } from 'zustand'
import * as api from '@/lib/graphApi'
import type {
  AgentResponse,
  DependencyResult,
  FileContent,
  GraphEdge,
  GraphNode,
  NodeDetail,
  Snapshot,
  Timeline,
  UiAction,
} from '@/lib/graphTypes'

/** One open code cell below the graph. */
export interface OpenFile {
  nodeId: string
  name: string
  path: string
  content: FileContent | null
  loading: boolean
  error: string | null
  collapsed: boolean
}

/** One turn in the Explorer's AI conversation. */
export interface AgentTurn {
  id: string
  role: 'user' | 'assistant'
  text: string
  intent?: string
  confidence?: number
  focusLabel?: string | null
  sources?: import('@/lib/types').Source[]
  pending?: boolean
  error?: string | null
}

interface GraphState {
  repository: string | null
  available: { repository: string; counts: Record<string, number> }[]

  /** Ancestors of the current view, root first — drives the breadcrumb. */
  path: GraphNode[]
  /** The node whose children are on screen. */
  focus: GraphNode | null
  nodes: GraphNode[]
  edges: GraphEdge[]
  truncated: boolean
  totalChildren: number

  selected: GraphNode | null
  detail: NodeDetail | null
  dependencies: DependencyResult | null

  /** Node IDs to emphasise, and the ones to dim, for dependency highlighting. */
  highlight: { dependencies: Set<string>; dependents: Set<string>; origin: string | null }

  openFiles: OpenFile[]

  timeline: Timeline | null
  /** Unix seconds. `null` means "now" — the live graph, not a snapshot. */
  cursor: number | null
  snapshot: Snapshot | null

  loading: boolean
  error: string | null
  counts: Record<string, number>

  /** Explorer-side AI conversation, separate from the Chat tab's history. */
  turns: AgentTurn[]
  asking: boolean
  ask: (question: string) => Promise<void>
  clearConversation: () => void

  bootstrap: () => Promise<void>
  selectRepository: (repository: string) => Promise<void>
  drillInto: (node: GraphNode) => Promise<void>
  navigateTo: (index: number) => Promise<void>
  goUp: () => Promise<void>
  selectNode: (node: GraphNode | null) => Promise<void>
  showDependencies: (nodeId: string) => Promise<void>
  clearHighlight: () => void
  openFile: (nodeId: string, name?: string, path?: string) => Promise<void>
  closeFile: (nodeId: string) => void
  toggleFileCollapsed: (nodeId: string) => void
  setCursor: (timestamp: number | null) => Promise<void>
  applyAgentAction: (action: UiAction | null) => Promise<void>
  focusById: (nodeId: string) => Promise<void>
}

const EMPTY_HIGHLIGHT = {
  dependencies: new Set<string>(),
  dependents: new Set<string>(),
  origin: null as string | null,
}

const message = (err: unknown) => (err instanceof Error ? err.message : String(err))
const uid = () => Math.random().toString(36).slice(2, 10)

export const useGraphStore = create<GraphState>((set, get) => ({
  repository: null,
  available: [],
  path: [],
  focus: null,
  nodes: [],
  edges: [],
  truncated: false,
  totalChildren: 0,
  selected: null,
  detail: null,
  dependencies: null,
  highlight: EMPTY_HIGHLIGHT,
  openFiles: [],
  timeline: null,
  cursor: null,
  snapshot: null,
  loading: false,
  error: null,
  counts: {},
  turns: [],
  asking: false,

  /**
   * Ask the LangGraph agent, then let its `ui_action` drive the graph — this
   * is what makes "what depends on this?" highlight rather than merely
   * describe.
   */
  async ask(question) {
    const trimmed = question.trim()
    const { repository, asking } = get()
    if (!trimmed || !repository || asking) return

    const pendingId = uid()
    set((s) => ({
      asking: true,
      turns: [
        ...s.turns,
        { id: uid(), role: 'user', text: trimmed },
        { id: pendingId, role: 'assistant', text: '', pending: true },
      ],
    }))

    const resolve = (patch: Partial<AgentTurn>) =>
      set((s) => ({
        asking: false,
        turns: s.turns.map((t) => (t.id === pendingId ? { ...t, pending: false, ...patch } : t)),
      }))

    try {
      const response: AgentResponse = await api.askAgent(trimmed, repository, `explorer:${repository}`)
      resolve({
        text: response.answer,
        intent: response.intent,
        confidence: response.confidence,
        focusLabel: response.focus?.label ?? null,
        sources: response.sources,
      })
      await get().applyAgentAction(response.ui_action)
    } catch (err) {
      resolve({ error: message(err) })
    }
  },

  clearConversation() {
    set({ turns: [] })
  },

  async bootstrap() {
    set({ loading: true, error: null })
    try {
      const { repositories } = await api.listGraphRepositories()
      set({ available: repositories, loading: false })
      const first = repositories[0]?.repository
      if (first && !get().repository) await get().selectRepository(first)
    } catch (err) {
      set({ loading: false, error: message(err) })
    }
  },

  async selectRepository(repository) {
    set({
      repository,
      loading: true,
      error: null,
      path: [],
      nodes: [],
      edges: [],
      selected: null,
      detail: null,
      dependencies: null,
      highlight: EMPTY_HIGHLIGHT,
      openFiles: [],
      cursor: null,
      snapshot: null,
      turns: [],
    })
    try {
      const [overview, timeline] = await Promise.all([
        api.fetchOverview(repository),
        // A repository with no git history still has a graph; the timeline
        // is optional decoration, so its failure must not blank the view.
        api.fetchTimeline(repository).catch(() => null),
      ])
      set({
        focus: overview.root,
        path: [overview.root],
        nodes: overview.nodes,
        edges: overview.edges,
        counts: overview.counts,
        timeline,
        truncated: false,
        totalChildren: overview.nodes.length,
        loading: false,
      })
    } catch (err) {
      set({ loading: false, error: message(err) })
    }
  },

  async drillInto(node) {
    // Files and symbols open rather than expand; only containers drill down.
    if (node.kind === 'file') {
      await get().openFile(node.id, node.name, node.key)
      await get().selectNode(node)
      return
    }
    if (node.kind === 'external') {
      await get().selectNode(node)
      return
    }

    set({ loading: true, error: null })
    try {
      const expansion = await api.expandNode(node.id)
      set((s) => ({
        focus: expansion.node,
        path: [...s.path, expansion.node],
        nodes: expansion.nodes,
        edges: expansion.edges,
        truncated: expansion.truncated,
        totalChildren: expansion.total_children,
        highlight: EMPTY_HIGHLIGHT,
        loading: false,
      }))
    } catch (err) {
      set({ loading: false, error: message(err) })
    }
  },

  async navigateTo(index) {
    const { path, repository } = get()
    const target = path[index]
    if (!target || !repository) return

    if (index === 0) {
      await get().selectRepository(repository)
      return
    }
    set({ loading: true, error: null })
    try {
      const expansion = await api.expandNode(target.id)
      set({
        focus: expansion.node,
        path: path.slice(0, index + 1),
        nodes: expansion.nodes,
        edges: expansion.edges,
        truncated: expansion.truncated,
        totalChildren: expansion.total_children,
        highlight: EMPTY_HIGHLIGHT,
        loading: false,
      })
    } catch (err) {
      set({ loading: false, error: message(err) })
    }
  },

  async goUp() {
    const { path } = get()
    if (path.length > 1) await get().navigateTo(path.length - 2)
  },

  async selectNode(node) {
    set({ selected: node, detail: null, dependencies: null })
    if (!node) return
    try {
      const detail = await api.fetchNodeDetail(node.id)
      // Discard if the user moved on while this was in flight.
      if (get().selected?.id === node.id) set({ detail })
    } catch (err) {
      set({ error: message(err) })
    }
  },

  async showDependencies(nodeId) {
    try {
      const result = await api.fetchDependencies(nodeId)
      set({
        dependencies: result,
        highlight: {
          dependencies: new Set(result.dependencies.map((n) => n.id)),
          dependents: new Set(result.dependents.map((n) => n.id)),
          origin: nodeId,
        },
      })
    } catch (err) {
      set({ error: message(err) })
    }
  },

  clearHighlight() {
    set({ highlight: EMPTY_HIGHLIGHT })
  },

  async openFile(nodeId, name, path) {
    const existing = get().openFiles.find((f) => f.nodeId === nodeId)
    if (existing) {
      // Already open: surface it rather than fetching twice.
      set((s) => ({
        openFiles: s.openFiles.map((f) =>
          f.nodeId === nodeId ? { ...f, collapsed: false } : f,
        ),
      }))
      return
    }

    set((s) => ({
      openFiles: [
        ...s.openFiles,
        {
          nodeId,
          name: name ?? nodeId.split('::').pop() ?? nodeId,
          path: path ?? '',
          content: null,
          loading: true,
          error: null,
          collapsed: false,
        },
      ],
    }))

    try {
      const content = await api.fetchFileContent(nodeId)
      set((s) => ({
        openFiles: s.openFiles.map((f) =>
          f.nodeId === nodeId
            ? {
                ...f,
                content,
                loading: false,
                name: content.node.name,
                path: content.relative_path,
              }
            : f,
        ),
      }))
    } catch (err) {
      set((s) => ({
        openFiles: s.openFiles.map((f) =>
          f.nodeId === nodeId ? { ...f, loading: false, error: message(err) } : f,
        ),
      }))
    }
  },

  closeFile(nodeId) {
    set((s) => ({ openFiles: s.openFiles.filter((f) => f.nodeId !== nodeId) }))
  },

  toggleFileCollapsed(nodeId) {
    set((s) => ({
      openFiles: s.openFiles.map((f) =>
        f.nodeId === nodeId ? { ...f, collapsed: !f.collapsed } : f,
      ),
    }))
  },

  async setCursor(timestamp) {
    const { repository } = get()
    if (!repository) return

    if (timestamp === null) {
      set({ cursor: null, snapshot: null })
      await get().selectRepository(repository)
      return
    }

    set({ cursor: timestamp, loading: true })
    try {
      const snapshot = await api.fetchSnapshot(repository, timestamp)
      // Time travel replaces the top level only; drilling into history would
      // need per-commit trees, which the backend deliberately does not build.
      set({
        snapshot,
        focus: snapshot.root,
        path: snapshot.root ? [snapshot.root] : [],
        nodes: snapshot.nodes,
        edges: snapshot.edges,
        truncated: false,
        totalChildren: snapshot.nodes.length,
        loading: false,
      })
    } catch (err) {
      set({ loading: false, error: message(err) })
    }
  },

  async applyAgentAction(action) {
    if (!action) return
    if (action.type === 'open_file') {
      await get().openFile(action.node_id)
      await get().focusById(action.node_id)
      return
    }
    if (action.type === 'highlight_dependencies') {
      await get().focusById(action.node_id)
      await get().showDependencies(action.node_id)
      return
    }
    if (action.type === 'show_structure') {
      const { repository } = get()
      if (repository) await get().selectRepository(repository)
    }
  },

  /** Select a node by id, loading its detail even if it is off screen. */
  async focusById(nodeId) {
    const known = get().nodes.find((n) => n.id === nodeId)
    if (known) {
      await get().selectNode(known)
      return
    }
    try {
      const detail = await api.fetchNodeDetail(nodeId)
      set({ selected: detail.node, detail })
    } catch (err) {
      set({ error: message(err) })
    }
  },
}))
