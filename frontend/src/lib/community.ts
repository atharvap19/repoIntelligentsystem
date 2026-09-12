/** Community detection for the knowledge graph (label propagation). */
import type { GraphEdge, GraphNode } from './types'

export interface Community {
  id: number
  label: string
  color: string
  size: number
}

export interface CommunityResult {
  membership: Map<string, number>
  communities: Community[]
}

/** Distinct, readable hues — cycled if there are more communities than colors. */
const PALETTE = [
  '#5b8ff0', // blue
  '#f2994a', // orange
  '#eb5757', // red
  '#56ccc4', // teal
  '#3fb950', // green
  '#f2c94c', // yellow
  '#bb8fce', // mauve
  '#f7a8b8', // pink
  '#a9856b', // brown
  '#9b9b9b', // grey
  '#7986cb', // indigo
  '#4dd0e1', // cyan
]

/**
 * Asynchronous label propagation.
 *
 * Cheap and deterministic-enough for a UI overview: each node repeatedly
 * adopts the label most common among its neighbours until labels stop
 * changing (or a small iteration cap is hit). No tuning parameters, unlike
 * modularity-based methods, which matters here since this runs client-side
 * on every graph the user loads.
 */
export function detectCommunities(nodes: GraphNode[], edges: GraphEdge[]): CommunityResult {
  const ids = nodes.map((n) => n.id)
  const present = new Set(ids)
  const neighbors = new Map<string, string[]>(ids.map((id) => [id, []]))
  const degree = new Map<string, number>(ids.map((id) => [id, 0]))

  for (const edge of edges) {
    if (!present.has(edge.source) || !present.has(edge.target) || edge.source === edge.target) continue
    neighbors.get(edge.source)!.push(edge.target)
    neighbors.get(edge.target)!.push(edge.source)
    degree.set(edge.source, (degree.get(edge.source) ?? 0) + 1)
    degree.set(edge.target, (degree.get(edge.target) ?? 0) + 1)
  }

  const label = new Map<string, string>(ids.map((id) => [id, id]))

  const order = [...ids]
  const maxIterations = 20
  for (let iter = 0; iter < maxIterations; iter++) {
    // Shuffled visiting order avoids the systematic bias plain LPA has when
    // nodes are always updated in the same sequence.
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[order[i], order[j]] = [order[j], order[i]]
    }

    let changed = 0
    for (const id of order) {
      const neigh = neighbors.get(id)!
      if (neigh.length === 0) continue

      const counts = new Map<string, number>()
      for (const n of neigh) {
        const l = label.get(n)!
        counts.set(l, (counts.get(l) ?? 0) + 1)
      }
      let best = label.get(id)!
      let bestCount = -1
      for (const [l, count] of counts) {
        if (count > bestCount || (count === bestCount && l < best)) {
          best = l
          bestCount = count
        }
      }
      if (best !== label.get(id)) {
        label.set(id, best)
        changed++
      }
    }
    if (changed === 0) break
  }

  // Collapse raw labels into small integer community ids, ordered by size
  // descending so the largest cluster gets a stable, low-index color.
  const groups = new Map<string, string[]>()
  for (const id of ids) {
    const l = label.get(id)!
    if (!groups.has(l)) groups.set(l, [])
    groups.get(l)!.push(id)
  }

  const byName = new Map(nodes.map((n) => [n.id, n.name]))
  const ordered = [...groups.entries()].sort((a, b) => b[1].length - a[1].length)

  const membership = new Map<string, number>()
  const communities: Community[] = ordered.map(([, members], index) => {
    members.forEach((id) => membership.set(id, index))
    const hub = members.reduce((best, id) =>
      (degree.get(id) ?? 0) > (degree.get(best) ?? 0) ? id : best,
    )
    return {
      id: index,
      label: byName.get(hub) ?? hub,
      color: PALETTE[index % PALETTE.length],
      size: members.length,
    }
  })

  return { membership, communities }
}
