import { create } from 'zustand'
import * as api from '@/lib/api'
import type {
  AnswerHighlight,
  GraphEdge,
  GraphNode,
  NodeKind,
  SelectionContext,
} from '@/lib/types'

const KNOWLEDGE_KINDS = new Set<NodeKind>(['file', 'class', 'function', 'method'])

/** The backend caps a node-set request at this many ids. */
const NODE_SET_LIMIT = 60

const message = (err: unknown) => (err instanceof Error ? err.message : String(err))

interface GraphState {
  repository: string | null
  nodes: GraphNode[]
  edges: GraphEdge[]
  counts: Record<string, number>
  totalNodes: number
  truncated: boolean
  /**
   * Bumped when a different graph is loaded, not when nodes are merged in.
   * Community colours key off it, so an answer that adds nodes does not
   * repaint the whole graph.
   */
  version: number
  loading: boolean
  error: string | null

  selectedId: string | null
  /** Nodes the latest answer drew on. Shown whenever nothing is selected. */
  highlight: AnswerHighlight | null
  /** Bumped to ask the canvas to re-frame onto whatever is emphasised. */
  frameRequest: number

  load: (repository: string) => Promise<void>
  reset: () => void
  select: (nodeId: string | null, frame?: boolean) => void
  /** Select a node that may not be loaded yet, then frame it. */
  focusNode: (nodeId: string) => Promise<void>
  /** Light up an answer's nodes, loading any the capped graph left out. */
  showAnswer: (highlight: AnswerHighlight) => Promise<void>
  clearEmphasis: () => void
  selectionContext: () => SelectionContext
}

const EMPTY = {
  nodes: [] as GraphNode[],
  edges: [] as GraphEdge[],
  counts: {},
  totalNodes: 0,
  truncated: false,
  selectedId: null,
  highlight: null,
  error: null,
}

export const useGraphStore = create<GraphState>((set, get) => {
  /**
   * Add nodes the knowledge graph withheld, so an answer about them has
   * something to light up.
   *
   * Containers come along too — a method arrives with its class and file —
   * because a symbol floating free of its source file says nothing about
   * where it lives. Up to three rounds covers method → class → file.
   */
  async function merge(ids: string[]) {
    let wanted = ids
    for (let round = 0; round < 3; round++) {
      const known = new Set(get().nodes.map((n) => n.id))
      const missing = [...new Set(wanted)].filter((id) => !known.has(id)).slice(0, NODE_SET_LIMIT)
      if (!missing.length) return

      const { nodes, edges } = await api.fetchNodeSet(missing)
      const added = nodes.filter((n) => KNOWLEDGE_KINDS.has(n.kind) && !known.has(n.id))
      if (!added.length) return

      const present = new Set([...known, ...added.map((n) => n.id)])
      const contains: GraphEdge[] = [...get().nodes, ...added]
        .filter((n) => n.kind !== 'file' && n.parent_id && present.has(n.parent_id))
        .filter((n) => added.some((a) => a.id === n.id || a.id === n.parent_id))
        .map((n) => ({ kind: 'CONTAINS', source: n.parent_id!, target: n.id }))

      set((s) => ({
        nodes: [...s.nodes, ...added],
        edges: [
          ...s.edges,
          ...contains,
          ...edges
            .filter((e) => present.has(e.source) && present.has(e.target))
            .map((e) => ({ kind: e.kind, source: e.source, target: e.target })),
        ],
      }))

      wanted = added
        .filter((n) => n.kind !== 'file' && n.parent_id && !present.has(n.parent_id))
        .map((n) => n.parent_id!)
    }
  }

  return {
    repository: null,
    ...EMPTY,
    version: 0,
    loading: false,
    frameRequest: 0,

    async load(repository) {
      set({ repository, ...EMPTY, loading: true })
      try {
        const graph = await api.fetchKnowledgeGraph(repository)
        // Discard if the user switched repositories while this was in flight.
        if (get().repository !== repository) return
        set((s) => ({
          nodes: graph.nodes,
          edges: graph.edges,
          counts: graph.counts,
          totalNodes: graph.total_nodes,
          truncated: graph.truncated,
          version: s.version + 1,
          loading: false,
        }))
      } catch (err) {
        if (get().repository === repository) set({ loading: false, error: message(err) })
      }
    },

    reset() {
      set((s) => ({ repository: null, ...EMPTY, loading: false, version: s.version + 1 }))
    },

    select(nodeId, frame = false) {
      set((s) => ({
        selectedId: nodeId,
        frameRequest: frame ? s.frameRequest + 1 : s.frameRequest,
      }))
    },

    async focusNode(nodeId) {
      try {
        await merge([nodeId])
      } catch (err) {
        set({ error: message(err) })
      }
      if (get().nodes.some((n) => n.id === nodeId)) get().select(nodeId, true)
    },

    async showAnswer(highlight) {
      if (!highlight.node_ids.length) {
        set({ highlight: null })
        return
      }
      try {
        await merge(highlight.node_ids)
      } catch {
        // Highlighting what is already loaded is still worth doing.
      }
      const loaded = new Set(get().nodes.map((n) => n.id))
      const node_ids = highlight.node_ids.filter((id) => loaded.has(id))
      set((s) => ({
        highlight: node_ids.length
          ? {
              node_ids,
              focus_node_id:
                highlight.focus_node_id && loaded.has(highlight.focus_node_id)
                  ? highlight.focus_node_id
                  : node_ids[0],
            }
          : null,
        // The answer is the newer statement of what matters; a selection
        // would otherwise hide it.
        selectedId: null,
        frameRequest: s.frameRequest + 1,
      }))
    },

    clearEmphasis() {
      set((s) => ({ selectedId: null, highlight: null, frameRequest: s.frameRequest + 1 }))
    },

    selectionContext() {
      const { repository, selectedId, nodes } = get()
      const node = selectedId ? nodes.find((n) => n.id === selectedId) : undefined
      if (!node) return { repository: repository ?? '' }
      return {
        repository: repository ?? '',
        file: node.kind === 'file' ? node.key : (node.data.relative_path ?? ''),
        symbol: node.kind === 'file' ? '' : node.name,
        node_id: node.id,
      }
    },
  }
})

export function useSelectedNode(): GraphNode | null {
  return useGraphStore((s) =>
    s.selectedId ? (s.nodes.find((n) => n.id === s.selectedId) ?? null) : null,
  )
}
