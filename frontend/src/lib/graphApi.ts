/**
 * Client for the Phase 3 graph, timeline and agent endpoints.
 *
 * Every call fetches one level. There is no "load the whole graph" call by
 * design — FastAPI's graph is 6,363 nodes, and the backend deliberately
 * refuses to serve it in one piece.
 */
import { ApiError } from './api'
import type {
  AgentResponse,
  DependencyResult,
  FileContent,
  GraphExpansion,
  GraphNode,
  GraphOverview,
  NodeDetail,
  Snapshot,
  Timeline,
} from './graphTypes'

const BASE = import.meta.env.VITE_API_BASE ?? '/api'

async function request<T>(path: string, timeoutMs = 30_000): Promise<T> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(`${BASE}${path}`, { signal: controller.signal })
    if (!response.ok) {
      let detail = `Request failed (${response.status})`
      try {
        const body = await response.json()
        if (typeof body?.detail === 'string') detail = body.detail
      } catch {
        /* non-JSON error body */
      }
      throw new ApiError(detail, response.status)
    }
    return (await response.json()) as T
  } catch (err) {
    if (err instanceof ApiError) throw err
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new ApiError('The graph request timed out.')
    }
    throw new ApiError('Could not reach the graph API.')
  } finally {
    clearTimeout(timer)
  }
}

const q = encodeURIComponent

export function listGraphRepositories() {
  return request<{ repositories: { repository: string; counts: Record<string, number> }[] }>(
    '/graph/repositories',
  )
}

export function fetchOverview(repository: string) {
  return request<GraphOverview>(`/graph/${q(repository)}/overview`)
}

export function expandNode(nodeId: string, limit = 120) {
  return request<GraphExpansion>(`/graph/node/expand?node_id=${q(nodeId)}&limit=${limit}`)
}

export function fetchNodeDetail(nodeId: string) {
  return request<NodeDetail>(`/graph/node?node_id=${q(nodeId)}`)
}

export function fetchDependencies(nodeId: string) {
  return request<DependencyResult>(`/graph/node/dependencies?node_id=${q(nodeId)}`)
}

export function fetchFileContent(nodeId: string) {
  // Large files take longer to read and redact than a metadata lookup.
  return request<FileContent>(`/graph/node/content?node_id=${q(nodeId)}`, 60_000)
}

export function fetchTimeline(repository: string) {
  return request<Timeline>(`/graph/${q(repository)}/timeline`)
}

export function fetchSnapshot(repository: string, timestamp: number) {
  return request<Snapshot>(`/graph/${q(repository)}/snapshot?timestamp=${timestamp}`)
}

export function searchGraph(repository: string, term: string, limit = 25) {
  return request<{ results: GraphNode[] }>(
    `/graph/${q(repository)}/search?q=${q(term)}&limit=${limit}`,
  )
}

/**
 * Ask the LangGraph agent. Generation runs locally on CPU and routinely takes
 * two to six minutes, hence the very long ceiling.
 */
export async function askAgent(
  question: string,
  repository: string,
  conversationId: string,
  topK = 5,
): Promise<AgentResponse> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), 900_000)
  try {
    const response = await fetch(`${BASE}/agent/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        repository,
        conversation_id: conversationId,
        top_k: topK,
      }),
      signal: controller.signal,
    })
    if (!response.ok) {
      let detail = `Request failed (${response.status})`
      try {
        const body = await response.json()
        if (typeof body?.detail === 'string') detail = body.detail
      } catch {
        /* non-JSON error body */
      }
      throw new ApiError(detail, response.status)
    }
    return (await response.json()) as AgentResponse
  } catch (err) {
    if (err instanceof ApiError) throw err
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new ApiError('The agent did not respond in time.')
    }
    throw new ApiError('Could not reach the agent.')
  } finally {
    clearTimeout(timer)
  }
}
