import { useMemo } from 'react'
import { create } from 'zustand'
import * as api from '@/lib/api'
import { ApiError } from '@/lib/api'
import { DEMO_REPOSITORIES } from '@/lib/demo'
import type {
  ConnectionState,
  Message,
  Repository,
  RunSettings,
  Source,
} from '@/lib/types'

const uid = () => Math.random().toString(36).slice(2, 10)

const DEFAULT_SETTINGS: RunSettings = {
  topK: 5,
  model: 'qwen3:4b',
  embedModel: 'nomic-embed-text',
  minRelevance: 0,
}

export type WorkspaceView = 'chat' | 'explorer'

interface AppState {
  /** Which workspace is on screen. Explorer is additive; chat is untouched. */
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
  importRepository: (url: string) => Promise<void>
  indexRepository: (id: string) => Promise<void>
  removeRepository: (id: string) => void
  ask: (question: string) => Promise<void>
  clearConversation: () => void
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

  selectRepository(id) {
    if (get().activeRepositoryId === id) return
    set({ activeRepositoryId: id, messages: [], focusedChunkKey: null })
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

    try {
      const response = await api.chat(trimmed, settings.topK, demo, activeRepository?.name)
      resolve({
        content: response.answer ?? '',
        sources: response.sources ?? [],
        timings: response.timings,
        latencyMs: performance.now() - startedAt,
      })
    } catch (err) {
      resolve({
        error:
          err instanceof ApiError
            ? err.message
            : err instanceof Error
              ? err.message
              : 'Something went wrong.',
        latencyMs: performance.now() - startedAt,
      })
    }
  },

  clearConversation() {
    set({ messages: [], focusedChunkKey: null })
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
