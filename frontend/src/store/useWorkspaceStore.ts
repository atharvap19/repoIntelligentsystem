import { create } from 'zustand'
import * as api from '@/lib/api'
import type { ConnectionState, RepositoryEntry, RepositoryStatus } from '@/lib/types'
import { useChatStore } from '@/store/useChatStore'
import { useGraphStore } from '@/store/useGraphStore'

const STORAGE_KEY = 'repoint.repository'
const POLL_MS = 1500

const isFinished = (status: RepositoryStatus) =>
  status.state === 'ready' || status.state === 'error'

function remember(name: string | null) {
  try {
    if (name) localStorage.setItem(STORAGE_KEY, name)
    else localStorage.removeItem(STORAGE_KEY)
  } catch {
    /* private mode */
  }
}

function remembered(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY)
  } catch {
    return null
  }
}

/** One poll loop per repository being imported. */
const polls = new Map<string, number>()

interface WorkspaceState {
  connection: ConnectionState
  repositories: RepositoryEntry[]
  active: string | null

  bootstrap: () => Promise<void>
  select: (repository: string) => void
  /** Returns an error message, or null once the import has started. */
  startImport: (url: string) => Promise<string | null>
}

export const useWorkspaceStore = create<WorkspaceState>((set, get) => {
  function upsert(status: RepositoryStatus, counts?: Record<string, number>) {
    set((s) => {
      const existing = s.repositories.find((r) => r.repository === status.repository)
      const entry: RepositoryEntry = {
        repository: status.repository,
        counts: counts ?? existing?.counts ?? {},
        status,
      }
      const repositories = existing
        ? s.repositories.map((r) => (r.repository === status.repository ? entry : r))
        : [...s.repositories, entry]
      return { repositories }
    })
  }

  /** Load the graph for the active repository as soon as it exists. */
  function loadGraphIfReady(repository: string) {
    const entry = get().repositories.find((r) => r.repository === repository)
    const graph = useGraphStore.getState()
    if (
      get().active === repository &&
      entry?.status.graph_ready &&
      (graph.repository !== repository || (!graph.nodes.length && !graph.loading))
    ) {
      void graph.load(repository)
    }
  }

  function poll(repository: string) {
    if (polls.has(repository)) return
    const tick = async () => {
      try {
        const status = await api.repositoryStatus(repository)
        upsert(status)
        loadGraphIfReady(repository)
        if (isFinished(status)) {
          polls.delete(repository)
          // Counts arrive with the listing, not the status.
          const entries = await api.listRepositories().catch(() => null)
          const entry = entries?.find((e) => e.repository === repository)
          if (entry) upsert(entry.status, entry.counts)
          return
        }
      } catch {
        // A blip while the backend is busy embedding; keep polling.
      }
      polls.set(repository, window.setTimeout(tick, POLL_MS))
    }
    polls.set(repository, window.setTimeout(tick, 0))
  }

  return {
    connection: 'checking',
    repositories: [],
    active: null,

    async bootstrap() {
      set({ connection: 'checking' })
      if (!(await api.health())) {
        set({ connection: 'offline' })
        return
      }
      set({ connection: 'online' })

      try {
        const repositories = await api.listRepositories()
        set({ repositories })
        repositories.filter((r) => !isFinished(r.status)).forEach((r) => poll(r.repository))

        const saved = remembered()
        const initial =
          repositories.find((r) => r.repository === saved) ??
          repositories.find((r) => r.status.graph_ready) ??
          repositories[0]
        if (initial) get().select(initial.repository)
      } catch {
        set({ connection: 'offline' })
      }
    },

    select(repository) {
      if (get().active === repository && useGraphStore.getState().repository === repository) return
      set({ active: repository })
      remember(repository)
      // A different codebase is a different conversation: "it" must not
      // carry over to a file that is no longer on screen.
      useChatStore.getState().clear()
      useGraphStore.getState().reset()
      loadGraphIfReady(repository)
    },

    async startImport(url) {
      try {
        const status = await api.importRepository(url)
        upsert(status)
        get().select(status.repository)
        if (isFinished(status)) {
          loadGraphIfReady(status.repository)
          const entries = await api.listRepositories().catch(() => null)
          const entry = entries?.find((e) => e.repository === status.repository)
          if (entry) upsert(entry.status, entry.counts)
        } else {
          poll(status.repository)
        }
        return null
      } catch (err) {
        return err instanceof Error ? err.message : String(err)
      }
    },
  }
})

export function useActiveRepository(): RepositoryEntry | null {
  return useWorkspaceStore(
    (s) => s.repositories.find((r) => r.repository === s.active) ?? null,
  )
}
