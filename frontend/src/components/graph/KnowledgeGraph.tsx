import { useEffect, useMemo, useRef, useState } from 'react'
import { Button } from '@/components/common/Button'
import { Tag } from '@/components/common/Bits'
import { IconClose, IconMinus, IconPlus, IconScan } from '@/components/common/Icons'
import { detectCommunities } from '@/lib/community'
import { cx, formatCount } from '@/lib/format'
import type { GraphNode } from '@/lib/types'
import { useChatStore } from '@/store/useChatStore'
import { useGraphStore, useSelectedNode } from '@/store/useGraphStore'
import { KnowledgeGraphRenderer, type Emphasis } from './renderer'

const KIND_LABEL: Record<GraphNode['kind'], string> = {
  file: 'file',
  class: 'class',
  function: 'function',
  method: 'method',
}

function location(node: GraphNode) {
  const path = node.kind === 'file' ? node.key : (node.data.relative_path ?? '')
  const lines =
    node.data.start_line && node.data.end_line ? `:${node.data.start_line}-${node.data.end_line}` : ''
  return `${path}${lines}`
}

/** What the selected node is, what it connects to, and what to ask about it. */
function SelectionCard() {
  const node = useSelectedNode()
  const edges = useGraphStore((s) => s.edges)
  const nodes = useGraphStore((s) => s.nodes)
  const select = useGraphStore((s) => s.select)
  const ask = useChatStore((s) => s.ask)
  const asking = useChatStore((s) => s.asking)
  const setDraft = useChatStore((s) => s.setDraft)

  const relations = useMemo(() => {
    if (!node) return null
    const byId = new Map(nodes.map((n) => [n.id, n]))
    const outgoing: Record<string, GraphNode[]> = {}
    const incoming: Record<string, GraphNode[]> = {}
    for (const edge of edges) {
      if (edge.kind === 'CONTAINS') continue
      if (edge.source === node.id && byId.has(edge.target)) {
        ;(outgoing[edge.kind] ??= []).push(byId.get(edge.target)!)
      } else if (edge.target === node.id && byId.has(edge.source)) {
        ;(incoming[edge.kind] ??= []).push(byId.get(edge.source)!)
      }
    }
    const children = nodes.filter((n) => n.parent_id === node.id)
    return { outgoing, incoming, children }
  }, [node, nodes, edges])

  if (!node || !relations) return null

  const rows: [string, GraphNode[]][] = [
    ['imports', relations.outgoing.IMPORTS ?? []],
    ['imported by', relations.incoming.IMPORTS ?? []],
    ['calls', relations.outgoing.CALLS ?? []],
    ['called by', relations.incoming.CALLS ?? []],
    ['inherits', relations.outgoing.INHERITS ?? []],
    ['inherited by', relations.incoming.INHERITS ?? []],
    ['contains', relations.children],
  ]

  return (
    <div className="absolute bottom-3 left-3 z-10 w-80 max-w-[calc(100%-1.5rem)] animate-fade-up rounded-lg border border-hairline bg-surface/95 shadow-xl backdrop-blur">
      <div className="flex items-start gap-2 border-b border-hairline px-3 py-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <Tag tone="accent">{KIND_LABEL[node.kind]}</Tag>
            <span className="truncate font-mono text-xs font-semibold text-ink">{node.name}</span>
          </div>
          <div className="mt-1 truncate font-mono text-2xs text-faint" title={location(node)}>
            {location(node)}
          </div>
        </div>
        <Button size="icon" onClick={() => select(null)} aria-label="Clear selection">
          <IconClose width={13} height={13} />
        </Button>
      </div>

      <div className="scrollbar-thin max-h-40 space-y-1 overflow-y-auto px-3 py-2">
        {rows.every(([, list]) => !list.length) && (
          <p className="text-2xs text-faint">No relationships to other nodes on screen.</p>
        )}
        {rows
          .filter(([, list]) => list.length)
          .map(([label, list]) => (
            <div key={label} className="flex gap-2 text-2xs">
              <span className="w-20 shrink-0 text-faint">
                {label} <span className="text-muted">{list.length}</span>
              </span>
              <span className="flex min-w-0 flex-wrap gap-x-1.5 gap-y-0.5">
                {list.slice(0, 8).map((other) => (
                  <button
                    key={other.id}
                    type="button"
                    onClick={() => select(other.id, true)}
                    className="max-w-full truncate font-mono text-muted hover:text-accent"
                  >
                    {other.name}
                  </button>
                ))}
                {list.length > 8 && <span className="text-faint">+{list.length - 8}</span>}
              </span>
            </div>
          ))}
      </div>

      <div className="flex gap-1.5 border-t border-hairline px-3 py-2">
        <Button
          size="sm"
          variant="outline"
          disabled={asking}
          onClick={() => void ask(`What does ${node.name} do?`)}
        >
          Explain this
        </Button>
        <Button
          size="sm"
          variant="outline"
          disabled={asking}
          onClick={() => void ask(`What depends on ${node.name}?`)}
        >
          What depends on it?
        </Button>
        <Button size="sm" onClick={() => setDraft(`About ${node.name}: `)}>
          Ask…
        </Button>
      </div>
    </div>
  )
}

export function KnowledgeGraph() {
  const nodes = useGraphStore((s) => s.nodes)
  const edges = useGraphStore((s) => s.edges)
  const version = useGraphStore((s) => s.version)
  const selectedId = useGraphStore((s) => s.selectedId)
  const highlight = useGraphStore((s) => s.highlight)
  const frameRequest = useGraphStore((s) => s.frameRequest)
  const select = useGraphStore((s) => s.select)

  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)
  const rendererRef = useRef<KnowledgeGraphRenderer | null>(null)
  const renderedVersion = useRef(-1)
  const selectedRef = useRef(selectedId)
  selectedRef.current = selectedId

  const [hovered, setHovered] = useState<GraphNode | null>(null)
  const [hidden, setHidden] = useState<Set<number>>(new Set())
  const [legendOpen, setLegendOpen] = useState(true)

  // Communities are detected once per loaded graph. Nodes merged in by an
  // answer take their parent's colour instead, so answering never repaints.
  const { communities, membership } = useMemo(
    () => detectCommunities(nodes, edges),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [version],
  )

  const adjacency = useMemo(() => {
    const map = new Map<string, Set<string>>()
    for (const edge of edges) {
      if (!map.has(edge.source)) map.set(edge.source, new Set())
      if (!map.has(edge.target)) map.set(edge.target, new Set())
      map.get(edge.source)!.add(edge.target)
      map.get(edge.target)!.add(edge.source)
    }
    return map
  }, [edges])

  const emphasis = useMemo<Emphasis | null>(() => {
    if (selectedId) {
      const neighbours = [...(adjacency.get(selectedId) ?? [])]
      return {
        ids: new Set([selectedId, ...neighbours]),
        focus: selectedId,
        mode: 'selection',
        order: [selectedId, ...neighbours],
      }
    }
    if (highlight) {
      return {
        ids: new Set(highlight.node_ids),
        focus: highlight.focus_node_id,
        mode: 'answer',
        order: highlight.node_ids,
      }
    }
    return null
  }, [selectedId, highlight, adjacency])

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const renderer = new KnowledgeGraphRenderer(canvas, {
      onSelect: (id) => select(id && id === selectedRef.current ? null : id),
      onOpen: (id) => select(id, true),
      onHover: (node, x, y) => {
        setHovered((current) => (current?.id === node?.id ? current : node))
        if (tooltipRef.current) tooltipRef.current.style.transform = `translate(${x + 14}px, ${y + 14}px)`
      },
    })
    rendererRef.current = renderer

    const resize = new ResizeObserver(([entry]) => {
      renderer.resize(entry.contentRect.width, entry.contentRect.height)
    })
    resize.observe(container)

    const theme = new MutationObserver(() => renderer.refreshTheme())
    theme.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })

    return () => {
      resize.disconnect()
      theme.disconnect()
      renderer.destroy()
      rendererRef.current = null
      renderedVersion.current = -1
    }
  }, [select])

  useEffect(() => {
    const fresh = renderedVersion.current !== version
    renderedVersion.current = version
    if (fresh) setHidden(new Set())
    rendererRef.current?.setGraph(
      nodes,
      edges,
      membership,
      communities.map((c) => c.color),
      fresh,
    )
  }, [nodes, edges, version, membership, communities])

  useEffect(() => rendererRef.current?.setEmphasis(emphasis), [emphasis])
  useEffect(() => rendererRef.current?.setHidden(hidden), [hidden])

  useEffect(() => {
    if (!frameRequest) return
    // Give merged nodes a moment to settle beside their parents first.
    const timer = window.setTimeout(() => rendererRef.current?.frameEmphasis(), 320)
    return () => window.clearTimeout(timer)
  }, [frameRequest])

  const toggleCommunity = (id: number) =>
    setHidden((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })

  return (
    <div ref={containerRef} className="relative h-full min-h-0 w-full overflow-hidden bg-canvas">
      <canvas ref={canvasRef} className="absolute inset-0 touch-none" style={{ cursor: 'grab' }} />

      <div
        ref={tooltipRef}
        className={cx(
          'pointer-events-none absolute left-0 top-0 z-20 max-w-xs rounded-md border border-hairline bg-surface/95 px-2 py-1.5 shadow-lg backdrop-blur',
          !hovered && 'hidden',
        )}
      >
        {hovered && (
          <>
            <div className="flex items-center gap-1.5">
              <span className="font-mono text-[0.58rem] uppercase text-faint">{hovered.kind}</span>
              <span className="truncate font-mono text-2xs font-medium text-ink">{hovered.name}</span>
            </div>
            <div className="truncate font-mono text-[0.62rem] text-muted">{location(hovered)}</div>
          </>
        )}
      </div>

      <div className="absolute right-3 top-3 z-10 flex flex-col gap-0.5 rounded-md border border-hairline bg-surface/90 p-0.5 shadow-sm backdrop-blur">
        <Button size="icon" title="Zoom in" aria-label="Zoom in" onClick={() => rendererRef.current?.zoomBy(1.35)}>
          <IconPlus width={13} height={13} />
        </Button>
        <Button size="icon" title="Zoom out" aria-label="Zoom out" onClick={() => rendererRef.current?.zoomBy(1 / 1.35)}>
          <IconMinus width={13} height={13} />
        </Button>
        <Button size="icon" title="Fit the whole graph" aria-label="Fit graph" onClick={() => rendererRef.current?.fitAll()}>
          <IconScan width={13} height={13} />
        </Button>
      </div>

      {communities.length > 0 && (
        <div className="absolute left-3 top-3 z-10 w-52 rounded-lg border border-hairline bg-surface/90 shadow-sm backdrop-blur">
          <button
            type="button"
            onClick={() => setLegendOpen((v) => !v)}
            className="flex w-full items-center justify-between px-3 py-2 text-2xs font-semibold uppercase tracking-[0.14em] text-faint hover:text-ink"
          >
            Communities
            <span className="font-mono normal-case tracking-normal">{communities.length}</span>
          </button>
          {legendOpen && (
            <>
              <div className="flex items-center gap-3 border-t border-hairline px-3 py-1.5 text-[0.62rem] text-faint">
                <span className="flex items-center gap-1">
                  <span className="h-2 w-2 rounded-full border border-faint" /> file
                </span>
                <span className="flex items-center gap-1">
                  <span className="h-1.5 w-1.5 rotate-45 bg-faint" /> class
                </span>
                <span className="flex items-center gap-1">
                  <span className="h-1.5 w-1.5 rounded-full bg-faint" /> function
                </span>
              </div>
              <label className="flex cursor-pointer items-center gap-2 border-t border-hairline px-3 py-1.5 text-2xs text-muted hover:bg-raised">
                <input
                  type="checkbox"
                  checked={hidden.size === 0}
                  onChange={() =>
                    setHidden(hidden.size === 0 ? new Set(communities.map((c) => c.id)) : new Set())
                  }
                  className="accent-accent"
                />
                Show all
              </label>
              <div className="scrollbar-thin max-h-56 overflow-y-auto border-t border-hairline py-1">
                {communities.map((c) => (
                  <label
                    key={c.id}
                    className={cx(
                      'flex cursor-pointer items-center gap-2 px-3 py-1 text-2xs hover:bg-raised',
                      hidden.has(c.id) ? 'text-faint' : 'text-ink',
                    )}
                  >
                    <input
                      type="checkbox"
                      checked={!hidden.has(c.id)}
                      onChange={() => toggleCommunity(c.id)}
                      className="accent-accent"
                    />
                    <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: c.color }} />
                    <span className="min-w-0 flex-1 truncate font-mono">{c.label}</span>
                    <span className="shrink-0 text-faint">{formatCount(c.size)}</span>
                  </label>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      <SelectionCard />
    </div>
  )
}
