import { create } from 'zustand'
import * as api from '@/lib/graphApi'
import type {
  CommitDetail,
  CommitTimeline,
  DependencyResult,
  ExplorerContextPayload,
  FileContent,
  GraphEdge,
  GraphNode,
  NavigationTarget,
  NodeDetail,
  RepositoryFlow,
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
  /** An ordered chain from a flow answer, drawn as a path through the graph. */
  flow: RepositoryFlow | null

  openFiles: OpenFile[]

  timeline: Timeline | null
  /** Commit dots (Part 11). Separate from `timeline`, which is module activity. */
  commits: CommitTimeline | null
  /** The commit being viewed, or null for the live repository. */
  commit: CommitDetail | null
  /** Unix seconds. `null` means "now" — the live graph, not a snapshot. */
  cursor: number | null
  snapshot: Snapshot | null

  loading: boolean
  error: string | null
  counts: Record<string, number>

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
  selectCommit: (sha: string | null) => Promise<void>
  /** Render a derived flow as an ordered chain across hierarchy levels. */
  showFlow: (startNodeId?: string) => Promise<void>
  applyAgentAction: (action: UiAction | null) => Promise<void>
  applyNavigation: (target: NavigationTarget) => Promise<void>
  revealNode: (nodeId: string) => Promise<void>
  focusById: (nodeId: string) => Promise<void>
  /** What the user currently has open, for the agent's context (Part 3). */
  explorerContext: () => ExplorerContextPayload
}

const EMPTY_HIGHLIGHT = {
  dependencies: new Set<string>(),
  dependents: new Set<string>(),
  origin: null as string | null,
}

const message = (err: unknown) => (err instanceof Error ? err.message : String(err))

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
  flow: null,
  openFiles: [],
  timeline: null,
  commits: null,
  commit: null,
  cursor: null,
  snapshot: null,
  loading: false,
  error: null,
  counts: {},

  /**
   * Describe the current selection for the agent.
   *
   * This is the whole of Part 3 on the client side: the question travels with
   * what the user is looking at, so "explain this" needs no antecedent in the
   * text. Both the Chat view and the Explorer sidebar read it — in Chat it is
   * usually near-empty, which is correct.
   */
  explorerContext() {
    const { repository, selected, path, openFiles, highlight, commit } = get()

    const file =
      selected?.kind === 'file'
        ? selected.key
        : (selected?.data.relative_path ?? openFiles[0]?.path ?? '')
    const symbol =
      selected && selected.kind !== 'file' && selected.kind !== 'module'
        ? selected.name
        : ''

    return {
      repository: repository ?? '',
      // The breadcrumb's module level, which is where the user actually is
      // even when nothing is selected.
      module: selected?.data.module ?? path[1]?.name ?? '',
      file,
      symbol,
      node_id: selected?.id ?? '',
      focus_node_ids: highlight.origin ? [highlight.origin] : [],
      commit_sha: commit?.commit.sha ?? '',
      timestamp: get().cursor,
    }
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
    if (get().repository === repository && get().focus) return

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
      flow: null,
      openFiles: [],
      cursor: null,
      snapshot: null,
      commit: null,
    })
    try {
      const [overview, timeline, commits] = await Promise.all([
        api.fetchOverview(repository),
        // A repository with no git history still has a graph; both history
        // views are optional decoration, so their failure must not blank the
        // structure.
        api.fetchTimeline(repository).catch(() => null),
        api.fetchCommits(repository).catch(() => null),
      ])
      set({
        focus: overview.root,
        path: [overview.root],
        nodes: overview.nodes,
        edges: overview.edges,
        counts: overview.counts,
        timeline,
        commits,
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
      set({ repository: null })
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
    set({ highlight: EMPTY_HIGHLIGHT, flow: null })
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
      set({ cursor: null, snapshot: null, commit: null, repository: null })
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

  /**
   * Move the repository to the state at a commit (Part 11).
   *
   * The commit detail and the snapshot are fetched together: the detail names
   * what changed and the snapshot is what the graph shows, and displaying one
   * without the other leaves the header describing a view that is not on
   * screen yet.
   */
  async selectCommit(sha) {
    const { repository } = get()
    if (!repository) return

    if (sha === null) {
      set({ commit: null, cursor: null, snapshot: null, repository: null })
      await get().selectRepository(repository)
      return
    }

    set({ loading: true, error: null })
    try {
      const [detail, snapshot] = await Promise.all([
        api.fetchCommit(repository, sha),
        api.fetchCommitSnapshot(repository, sha),
      ])
      set({
        commit: detail,
        cursor: detail.commit.authored_at,
        snapshot,
        focus: snapshot.root,
        path: snapshot.root ? [snapshot.root] : [],
        nodes: snapshot.nodes,
        edges: snapshot.edges,
        truncated: false,
        totalChildren: snapshot.nodes.length,
        highlight: EMPTY_HIGHLIGHT,
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
    if (action.type === 'show_flow') {
      await get().showFlow(action.node_ids[0])
      return
    }
    if (action.type === 'show_structure') {
      const { repository } = get()
      if (repository) {
        set({ repository: null })
        await get().selectRepository(repository)
      }
    }
  },

  /**
   * Act on a structured navigation target — what [Explore this] does.
   *
   * Switching to the Explorer view is the caller's job, not this store's: the
   * view flag lives in the app store, and reaching across for it here would
   * make the two stores mutually dependent.
   */
  async applyNavigation(target) {
    if (!target.available) return
    const { repository } = get()

    if (target.repository && target.repository !== repository) {
      await get().selectRepository(target.repository)
    }

    if (target.focus_mode === 'flow') {
      await get().showFlow(target.node_id)
      return
    }

    if (target.focus_mode === 'timeline') {
      if (target.commit_sha) await get().selectCommit(target.commit_sha)
      // Otherwise the timeline is already on screen; the file below is what
      // gives it something to be about.
      if (target.node_id) await get().revealNode(target.node_id)
      return
    }

    if (target.target_type === 'architecture' || target.target_type === 'repository') {
      const name = target.repository || repository
      if (name) {
        set({ repository: null })
        await get().selectRepository(name)
      }
      return
    }

    if (!target.node_id) return

    await get().revealNode(target.node_id)

    if (target.focus_mode === 'neighborhood') {
      await get().showDependencies(target.node_id)
    }
    if (target.target_type === 'file' && target.file) {
      await get().openFile(target.node_id, target.file.split('/').pop(), target.file)
    }
  },

  /**
   * Draw a derived flow as its own view.
   *
   * A flow cannot be shown by expanding a parent, which is how every other
   * view here works. Its nodes deliberately cross levels — for a FastAPI-style
   * layout the chain runs `backend/main.py -> backend/routes/auth.py ->
   * backend/services/auth_service.py`, three different parents — so expanding
   * the common ancestor renders the first two files and silently drops the
   * rest. The set is fetched by name instead.
   *
   * The flow is re-fetched rather than rebuilt from the navigation target
   * because the target is flat, and the step *depths* are what the numbering
   * has to be truthful about.
   */
  async showFlow(startNodeId) {
    const { repository } = get()
    if (!repository) return

    set({ loading: true, error: null })
    try {
      const flow = await api.fetchFlow(repository, startNodeId || undefined)
      if (!flow.node_ids.length) {
        set({ loading: false })
        return
      }

      const [{ nodes, edges }, { path }] = await Promise.all([
        api.fetchNodeSet(flow.node_ids),
        // The breadcrumb still needs somewhere to be, and the layout needs a
        // container card to hang the chain beneath.
        api.fetchNodePath(flow.node_ids[0]),
      ])

      const container = path[path.length - 2] ?? path[0] ?? null
      set({
        flow,
        focus: container,
        path: container ? path.slice(0, path.length - 1) : [],
        nodes,
        edges,
        truncated: false,
        totalChildren: nodes.length,
        highlight: EMPTY_HIGHLIGHT,
        loading: false,
      })
    } catch (err) {
      set({ loading: false, error: message(err) })
    }
  },

  /**
   * Bring a node on screen wherever it sits in the hierarchy.
   *
   * The Explorer renders one level at a time, so selecting a deep node is not
   * enough — its parent has to be the level on screen and the breadcrumb has
   * to match, or the user lands somewhere with no way back up. The ancestor
   * chain comes from the backend in one call rather than by walking parents.
   */
  async revealNode(nodeId) {
    set({ loading: true, error: null })
    try {
      const { path } = await api.fetchNodePath(nodeId)
      const target = path[path.length - 1]
      if (!target) {
        set({ loading: false })
        return
      }

      // Symbols are not laid out as their own level; show the file that holds
      // them and select the symbol inside it.
      const container = [...path]
        .reverse()
        .find((n) => n.kind !== 'file' && n.id !== target.id)
      const parent = container ?? path[path.length - 2]

      if (parent) {
        const expansion = await api.expandNode(parent.id)
        const index = path.findIndex((n) => n.id === parent.id)
        set({
          focus: expansion.node,
          path: path.slice(0, index + 1),
          nodes: expansion.nodes,
          edges: expansion.edges,
          truncated: expansion.truncated,
          totalChildren: expansion.total_children,
        })
      }

      set({ loading: false })
      await get().selectNode(target)
    } catch (err) {
      set({ loading: false, error: message(err) })
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
