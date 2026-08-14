import { useMemo } from 'react'
import { create } from 'zustand'
import * as api from '@/lib/api'
import { ApiError } from '@/lib/api'
import * as graphApi from '@/lib/graphApi'
import { DEMO_REPOSITORIES } from '@/lib/demo'
import type { NavigationTarget } from '@/lib/graphTypes'
import { useGraphStore } from '@/store/useGraphStore'
import type {
  ConnectionState,
  Message,
  Repository,
  RunSettings,
  Source,
} from '@/lib/types'

const uid = () => Math.random().toString(36).slice(2, 10)

const message = (err: unknown) =>
  err instanceof ApiError || err instanceof Error ? err.message : 'Something went wrong.'

const DEFAULT_SETTINGS: RunSettings = {
  topK: 5,
  model: 'qwen3:4b',
  embedModel: 'nomic-embed-text',
  minRelevance: 0,
}

export type WorkspaceView = 'chat' | 'explorer'

/**
 * One conversation, two views.
 *
 * Phase 3 had a second conversation living in the graph store, so switching to
 * Explorer silently started over. Part 4 forbids that: there is one thread,
 * and the view is only where it is rendered. Everything conversational
 * therefore lives here, and the graph store keeps graph state only.
 *
 * The thread is keyed by repository — asking about a different codebase is a
 * different conversation, and carrying "it" across that boundary would resolve
 * to the wrong file.
 */
const conversationId = (repository: string | null) => `repoint:${repository ?? 'none'}`

interface AppState {
  /** Which workspace is on screen. The conversation is shared between them. */
  view: WorkspaceView
  setView: (view: WorkspaceView) => void

  connection: ConnectionState
  /** True when answers come from fixtures — either pinned or auto-fallback. */
  demo: boolean
  demoPinned: boolean

  repositories: Repository[]
  activeRepositoryId: string | null

  messages: Message[]
  asking: boolean

  settings: RunSettings
  /** Chunk currently expanded/focused in the Inspector, keyed file:chunk_index. */
  focusedChunkKey: string | null
  inspectorOpen: boolean
  railOpen: boolean

  bootstrap: () => Promise<void>
  setDemoPinned: (pinned: boolean) => void
  selectRepository: (id: string) => void
  /** Select by repository name — what the Explorer's picker calls. */
  selectRepositoryByName: (name: string) => void
  importRepository: (url: string) => Promise<void>
  indexRepository: (id: string) => Promise<void>
  removeRepository: (id: string) => void
  ask: (question: string) => Promise<void>
  clearConversation: () => void
  /** Follow an answer's [Explore this] into the graph, keeping the thread. */
  exploreFrom: (target: NavigationTarget) => Promise<void>
  updateSettings: (patch: Partial<RunSettings>) => void
  focusChunk: (key: string | null) => void
  toggleInspector: () => void
  toggleRail: () => void
}

export const chunkKey = (s: Source) => `${s.file}#${s.chunk_index ?? 0}`

export const useAppStore = create<AppState>((set, get) => ({
  view: 'chat',
  setView: (view) => set({ view }),

  connection: 'checking',
  demo: false,
  demoPinned: false,

  repositories: [],
  activeRepositoryId: null,

  messages: [],
  asking: false,

  settings: DEFAULT_SETTINGS,
  focusedChunkKey: null,
  inspectorOpen: true,
  railOpen: true,

  async bootstrap() {
    const alive = await api.health()

    if (alive) {
      const demo = get().demoPinned
      set({ connection: 'online', demo })

      // Adopt whatever the backend already has indexed, so a fresh page load
      // can query immediately instead of showing an empty workspace.
      try {
        const names = await api.listRepositories(demo)
        if (names.length) {
          const existing = get().repositories
          const discovered: Repository[] = names
            .filter((name) => !existing.some((r) => r.name === name))
            .map((name) => ({
              id: name,
              name,
              state: 'ready' as const,
              importedAt: Date.now(),
            }))
          set((s) => ({
            repositories: [...s.repositories, ...discovered],
            activeRepositoryId: s.activeRepositoryId ?? discovered[0]?.id ?? null,
          }))
        }
      } catch {
        // Listing is best-effort; the workspace still works via import.
      }
      return
    }

    set({
      connection: 'demo',
      demo: true,
      repositories: DEMO_REPOSITORIES,
      activeRepositoryId: DEMO_REPOSITORIES[0]?.id ?? null,
    })
  },

  setDemoPinned(pinned) {
    const { connection } = get()
    if (pinned) {
      const hasRepos = get().repositories.length > 0
      set({
        demoPinned: true,
        demo: true,
        repositories: hasRepos ? get().repositories : DEMO_REPOSITORIES,
        activeRepositoryId: get().activeRepositoryId ?? DEMO_REPOSITORIES[0]?.id ?? null,
      })
      return
    }
    // Un-pinning only returns to live mode if the backend is actually up.
    set({ demoPinned: false, demo: connection !== 'online' })
  },

  /**
   * Switch repositories in both views at once.
   *
   * The conversation resets, deliberately: the thread is keyed by repository
   * on the backend too, so carrying it across would leave "it" pointing at a
   * file in a codebase that is no longer open.
   */
  selectRepository(id) {
    if (get().activeRepositoryId === id) return
    set({ activeRepositoryId: id, messages: [], focusedChunkKey: null })

    const name = get().repositories.find((r) => r.id === id)?.name
    if (name && !get().demo) void useGraphStore.getState().selectRepository(name)
  },

  selectRepositoryByName(name) {
    const match = get().repositories.find((r) => r.name === name)
    if (match) {
      get().selectRepository(match.id)
      return
    }
    // Present in the graph but never imported through this session — adopt it
    // so both views agree on what is selected.
    const id = uid()
    set((s) => ({
      repositories: [...s.repositories, { id, name, state: 'ready', importedAt: Date.now() }],
      activeRepositoryId: id,
      messages: [],
      focusedChunkKey: null,
    }))
    void useGraphStore.getState().selectRepository(name)
  },

  async importRepository(url) {
    const id = uid()
    const provisionalName =
      url.replace(/\/+$/, '').split('/').pop()?.replace(/\.git$/, '') || 'repository'

    set((s) => ({
      repositories: [
        ...s.repositories,
        { id, name: provisionalName, url, state: 'cloning', importedAt: Date.now() },
      ],
      activeRepositoryId: id,
      messages: [],
    }))

    const patch = (fields: Partial<Repository>) =>
      set((s) => ({
        repositories: s.repositories.map((r) => (r.id === id ? { ...r, ...fields } : r)),
      }))

    try {
      const result = await api.importRepository(url, get().demo)
      if (result.status === 'error') {
        patch({ state: 'error', error: result.message })
        return
      }
      patch({
        state: 'unindexed',
        name: result.repository ?? provisionalName,
        path: result.path,
      })
    } catch (err) {
      patch({ state: 'error', error: err instanceof Error ? err.message : String(err) })
    }
  },

  async indexRepository(id) {
    const repo = get().repositories.find((r) => r.id === id)
    if (!repo?.path || repo.state === 'indexing') return

    const patch = (fields: Partial<Repository>) =>
      set((s) => ({
        repositories: s.repositories.map((r) => (r.id === id ? { ...r, ...fields } : r)),
      }))

    patch({ state: 'indexing', error: undefined })

    try {
      const stats = await api.indexRepository(repo.path, get().demo)
      patch({ state: 'ready', stats })
    } catch (err) {
      patch({ state: 'error', error: err instanceof Error ? err.message : String(err) })
    }
  },

  removeRepository(id) {
    set((s) => {
      const repositories = s.repositories.filter((r) => r.id !== id)
      const wasActive = s.activeRepositoryId === id
      return {
        repositories,
        activeRepositoryId: wasActive ? (repositories[0]?.id ?? null) : s.activeRepositoryId,
        messages: wasActive ? [] : s.messages,
      }
    })
  },

  /**
   * Ask the repository agent. The same call backs both views.
   *
   * The request carries whatever Explorer has selected, so a question asked
   * from the sidebar needs no antecedent — "what depends on this?" resolves
   * server-side against the selection. From the Chat view that payload is
   * near-empty, which is the correct description of what the user is looking
   * at.
   */
  async ask(question) {
    const trimmed = question.trim()
    if (!trimmed || get().asking) return

    const { settings, demo, activeRepositoryId, repositories } = get()
    const activeRepository = repositories.find((r) => r.id === activeRepositoryId)
    const pendingId = uid()

    set((s) => ({
      asking: true,
      focusedChunkKey: null,
      messages: [
        ...s.messages,
        {
          id: uid(),
          role: 'user',
          content: trimmed,
          repositoryId: activeRepositoryId ?? undefined,
          createdAt: Date.now(),
        },
        {
          id: pendingId,
          role: 'assistant',
          content: '',
          pending: true,
          repositoryId: activeRepositoryId ?? undefined,
          createdAt: Date.now(),
        },
      ],
    }))

    const resolve = (fields: Partial<Message>) =>
      set((s) => ({
        asking: false,
        messages: s.messages.map((m) =>
          m.id === pendingId ? { ...m, pending: false, ...fields } : m,
        ),
      }))

    const startedAt = performance.now()

    // Fixtures have no agent behind them, so demo mode keeps the Phase 2
    // route and simply offers no navigation.
    if (demo) {
      try {
        const response = await api.chat(trimmed, settings.topK, true, activeRepository?.name)
        resolve({
          content: response.answer ?? '',
          sources: response.sources ?? [],
          timings: response.timings,
          latencyMs: performance.now() - startedAt,
        })
      } catch (err) {
        resolve({ error: message(err), latencyMs: performance.now() - startedAt })
      }
      return
    }

    const graph = useGraphStore.getState()
    const repositoryName = activeRepository?.name ?? graph.repository ?? ''

    try {
      const response = await graphApi.askAgent(
        trimmed,
        repositoryName,
        conversationId(repositoryName),
        settings.topK,
        graph.explorerContext(),
      )

      resolve({
        content: response.answer ?? '',
        sources: response.sources ?? [],
        intent: response.intent,
        confidence: response.confidence,
        focusLabel: response.focus?.label ?? null,
        navigation: response.navigation,
        uiAction: response.ui_action,
        contextStats: response.context_stats,
        latencyMs: performance.now() - startedAt,
      })

      // Already in Explorer: apply the graph action immediately, because the
      // user can see the result. From the Chat view nothing moves until they
      // choose to explore — Part 5's "never force it".
      if (get().view === 'explorer' && response.ui_action) {
        await useGraphStore.getState().applyAgentAction(response.ui_action)
      }
    } catch (err) {
      resolve({ error: message(err), latencyMs: performance.now() - startedAt })
    }
  },

  clearConversation() {
    set({ messages: [], focusedChunkKey: null })
  },

  async exploreFrom(target) {
    if (!target?.available) return
    // Switch first so the graph work is visible while it happens rather than
    // landing the user on a finished view with no sense of where they went.
    set({ view: 'explorer' })
    await useGraphStore.getState().applyNavigation(target)
  },

  updateSettings(patch) {
    set((s) => ({ settings: { ...s.settings, ...patch } }))
  },

  focusChunk(key) {
    set({ focusedChunkKey: key })
  },

  toggleInspector() {
    set((s) => ({ inspectorOpen: !s.inspectorOpen }))
  },

  toggleRail() {
    set((s) => ({ railOpen: !s.railOpen }))
  },
}))

/**
 * Sources of the most recent answered turn — what the Inspector renders.
 * Selects the (stable) messages array and derives, so the hook does not return
 * a fresh object on every unrelated store update.
 */
export function useLatestSources(): { sources: Source[]; messageId: string | null } {
  const messages = useAppStore((s) => s.messages)
  return useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i]
      if (m.role === 'assistant' && m.sources?.length) {
        return { sources: m.sources, messageId: m.id }
      }
    }
    return EMPTY_SOURCES
  }, [messages])
}

const EMPTY_SOURCES: { sources: Source[]; messageId: string | null } = {
  sources: [],
  messageId: null,
}

export function useActiveRepository(): Repository | null {
  return useAppStore((s) => s.repositories.find((r) => r.id === s.activeRepositoryId) ?? null)
}
