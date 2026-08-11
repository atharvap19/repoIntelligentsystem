/**
 * Typed client for the FastAPI backend.
 *
 * Requests go to `/api/*`, which Vite proxies to the backend in dev (see
 * vite.config.ts) — that sidesteps the missing CORS middleware. Endpoints the
 * backend has not grown yet are marked below and degrade instead of throwing.
 */
import { DEMO_REPOSITORIES, demoChat, demoImport, demoIndex } from './demo'
import type { ChatResponse, ImportResponse, IndexStats } from './types'

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
  const { timeoutMs = 120_000, ...rest } = init ?? {}
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
      throw new ApiError('The backend did not respond in time. Is Ollama running?')
    }
    throw new ApiError('Could not reach the backend on :8000.')
  } finally {
    clearTimeout(timer)
  }
}

/** GET / — cheap liveness probe used to decide between live and demo mode. */
export async function health(): Promise<boolean> {
  try {
    await request<{ status: string }>('/', { method: 'GET', timeoutMs: 4000 })
    return true
  } catch {
    return false
  }
}

export async function chat(
  question: string,
  topK: number,
  demo: boolean,
  repository?: string,
): Promise<ChatResponse> {
  if (demo) return demoChat(question, topK)
  // `repository` is required as soon as more than one is indexed — the backend
  // only auto-resolves when exactly one exists.
  // qwen3 emits reasoning tokens before its answer, so a local round trip runs
  // 60-120s on CPU. Generous ceiling; the UI shows a pending state throughout.
  return request<ChatResponse>('/chat/', {
    method: 'POST',
    body: JSON.stringify({ question, top_k: topK, repository: repository ?? null }),
    timeoutMs: 600_000,
  })
}

/** GET /chat/repositories — what the backend can currently answer about. */
export async function listRepositories(demo: boolean): Promise<string[]> {
  if (demo) return DEMO_REPOSITORIES.map((r) => r.name)
  const body = await request<{ repositories: string[] }>('/chat/repositories', {
    method: 'GET',
    timeoutMs: 10_000,
  })
  return body.repositories ?? []
}

export async function importRepository(url: string, demo: boolean): Promise<ImportResponse> {
  if (demo) return demoImport(url)
  return request<ImportResponse>('/github/import', {
    method: 'POST',
    body: JSON.stringify({ url }),
    timeoutMs: 300_000,
  })
}

/**
 * Builds a repository's vector index. Runs for minutes on a large checkout —
 * the backend does this synchronously, hence the very long timeout.
 */
export async function indexRepository(path: string, demo: boolean): Promise<IndexStats> {
  if (demo) return demoIndex()
  try {
    return await request<IndexStats>('/index/', {
      method: 'POST',
      body: JSON.stringify({ path }),
      timeoutMs: 900_000,
    })
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) {
      throw new ApiError(`No checkout found at ${path}. Import the repository first.`, 404)
    }
    throw err
  }
}
