/**
 * Typed client for the FastAPI backend.
 *
 * Requests go to `/api/*`, which Vite proxies to the backend in dev (see
 * vite.config.ts) — that sidesteps the missing CORS middleware.
 */
import type {
  AgentResponse,
  GraphEdge,
  GraphNode,
  KnowledgeGraph,
  RepositoryEntry,
  RepositoryStatus,
  SelectionContext,
} from './types'

const BASE = import.meta.env.VITE_API_BASE ?? '/api'

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit & { timeoutMs?: number }): Promise<T> {
  const { timeoutMs = 30_000, ...rest } = init ?? {}
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)

  try {
    const response = await fetch(`${BASE}${path}`, {
      ...rest,
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json', ...rest.headers },
    })

    if (!response.ok) {
      // FastAPI puts validation and HTTPException text under `detail`.
      let detail = `Request failed (${response.status})`
      try {
        const body = await response.json()
        if (typeof body?.detail === 'string') detail = body.detail
        else if (Array.isArray(body?.detail)) detail = body.detail[0]?.msg ?? detail
      } catch {
        /* non-JSON error body */
      }
      throw new ApiError(detail, response.status)
    }

    return (await response.json()) as T
  } catch (err) {
    if (err instanceof ApiError) throw err
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new ApiError('The backend did not respond in time.')
    }
    throw new ApiError('Could not reach the backend on :8000.')
  } finally {
    clearTimeout(timer)
  }
}

const q = encodeURIComponent

/** GET / — cheap liveness probe. */
export async function health(): Promise<boolean> {
  try {
    await request<{ status: string }>('/', { timeoutMs: 4000 })
    return true
  } catch {
    return false
  }
}

export async function listRepositories(): Promise<RepositoryEntry[]> {
  const body = await request<{ repositories: RepositoryEntry[] }>('/repositories')
  return body.repositories ?? []
}

/** Starts clone → graph → code index. Returns at once with the job. */
export function importRepository(url: string) {
  return request<RepositoryStatus>('/repositories/import', {
    method: 'POST',
    body: JSON.stringify({ url }),
  })
}

export function repositoryStatus(repository: string) {
  return request<RepositoryStatus>(`/repositories/${q(repository)}/status`, { timeoutMs: 10_000 })
}

export function fetchKnowledgeGraph(repository: string) {
  return request<KnowledgeGraph>(`/graph/${q(repository)}/knowledge`, { timeoutMs: 60_000 })
}

/** A named set of nodes and the edges among them — capped at 60 by the backend. */
export function fetchNodeSet(nodeIds: string[]) {
  const params = nodeIds.map((id) => `id=${q(id)}`).join('&')
  return request<{ nodes: GraphNode[]; edges: GraphEdge[] }>(`/graph/nodes?${params}`)
}

export function searchGraph(repository: string, term: string, limit = 25) {
  return request<{ results: GraphNode[] }>(
    `/graph/${q(repository)}/search?q=${q(term)}&limit=${limit}`,
  )
}

/**
 * Ask the repository agent. Generation runs locally and routinely takes
 * minutes on CPU, hence the very long ceiling.
 */
export function askAgent(
  question: string,
  repository: string,
  conversationId: string,
  selection: SelectionContext,
) {
  return request<AgentResponse>('/agent/ask', {
    method: 'POST',
    body: JSON.stringify({
      question,
      repository,
      conversation_id: conversationId,
      explorer_context: selection,
    }),
    timeoutMs: 900_000,
  })
}
