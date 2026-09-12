import { useEffect, useMemo, useRef, useState } from 'react'
import { Button } from '@/components/common/Button'
import { StatusDot, Tag } from '@/components/common/Bits'
import { IconCheck, IconSearch, IconWarn } from '@/components/common/Icons'
import { KnowledgeGraph } from '@/components/graph/KnowledgeGraph'
import * as api from '@/lib/api'
import { cx, formatCount } from '@/lib/format'
import type { GraphNode, RepositoryEntry } from '@/lib/types'
import { useGraphStore } from '@/store/useGraphStore'
import { useActiveRepository } from '@/store/useWorkspaceStore'

const SEARCHABLE = new Set(['file', 'class', 'function', 'method'])

const STEPS = [
  { state: 'cloning', label: 'Clone the repository' },
  { state: 'building_graph', label: 'Build the knowledge graph' },
  { state: 'indexing_code', label: 'Index code for answers' },
] as const

/** Shown until the graph exists: which step the import is on. */
function BuildProgress({ entry }: { entry: RepositoryEntry }) {
  const { status } = entry
  const current = STEPS.findIndex((s) => s.state === status.state)
  const failed = status.state === 'error'

  return (
    <div className="flex h-full items-center justify-center p-8">
      <div className="w-full max-w-sm">
        <h2 className="font-mono text-sm font-semibold text-ink">{entry.repository}</h2>
        <ol className="mt-4 space-y-2.5">
          {STEPS.map((step, index) => {
            const done = !failed && current > index
            const active = !failed && current === index
            return (
              <li key={step.state} className="flex items-center gap-2.5 text-xs">
                <span className="grid h-4 w-4 place-items-center">
                  {done ? (
                    <IconCheck width={13} height={13} className="text-signal" />
                  ) : (
                    <StatusDot tone={active ? 'accent' : 'faint'} pulse={active} />
                  )}
                </span>
                <span className={cx(active ? 'text-ink' : done ? 'text-muted' : 'text-faint')}>
                  {step.label}
                </span>
              </li>
            )
          })}
        </ol>
        {failed && (
          <p className="mt-4 flex gap-1.5 text-xs leading-relaxed text-danger">
            <IconWarn width={13} height={13} className="mt-0.5 shrink-0" />
            {status.message || 'The import failed.'}
          </p>
        )}
        {!failed && (
          <p className="mt-4 text-2xs leading-relaxed text-faint">
            The graph appears as soon as it is built — usually seconds after cloning.
          </p>
        )}
      </div>
    </div>
  )
}

function GraphSearch() {
  const repository = useGraphStore((s) => s.repository)
  const nodes = useGraphStore((s) => s.nodes)
  const focusNode = useGraphStore((s) => s.focusNode)
  const [term, setTerm] = useState('')
  const [remote, setRemote] = useState<GraphNode[]>([])
  const [open, setOpen] = useState(false)
  const timer = useRef<number | null>(null)

  const local = useMemo(() => {
    const needle = term.trim().toLowerCase()
    if (needle.length < 2) return []
    // Exact name, then prefix, then anywhere in the name, then path-only
    // matches — so "_client.py" finds the file before the classes inside it.
    const rank = (n: GraphNode) => {
      const name = n.name.toLowerCase()
      return name === needle ? 0 : name.startsWith(needle) ? 1 : name.includes(needle) ? 2 : 3
    }
    return nodes
      .filter((n) => n.name.toLowerCase().includes(needle) || n.key.toLowerCase().includes(needle))
      .sort((a, b) => rank(a) - rank(b) || a.name.length - b.name.length)
      .slice(0, 12)
  }, [term, nodes])

  // The graph on screen may be capped; the backend search finds the rest,
  // and choosing one of those loads it into the graph.
  useEffect(() => {
    if (timer.current) window.clearTimeout(timer.current)
    if (!repository || term.trim().length < 2) {
      setRemote([])
      return
    }
    timer.current = window.setTimeout(async () => {
      try {
        const { results } = await api.searchGraph(repository, term.trim(), 20)
        setRemote(results.filter((n) => SEARCHABLE.has(n.kind)))
      } catch {
        setRemote([])
      }
    }, 250)
    return () => {
      if (timer.current) window.clearTimeout(timer.current)
    }
  }, [term, repository])

  const results = useMemo(() => {
    const seen = new Set(local.map((n) => n.id))
    return [...local, ...remote.filter((n) => !seen.has(n.id))].slice(0, 16)
  }, [local, remote])

  return (
    <div className="relative w-64">
      <IconSearch
        width={13}
        height={13}
        className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 text-faint"
      />
      <input
        value={term}
        onChange={(e) => {
          setTerm(e.target.value)
          setOpen(true)
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => window.setTimeout(() => setOpen(false), 150)}
        placeholder="Find a file or symbol…"
        spellCheck={false}
        className="h-7 w-full rounded-md border border-hairline bg-canvas pl-7 pr-2 font-mono text-2xs text-ink placeholder:text-faint focus:border-accent/60"
      />
      {open && results.length > 0 && (
        <div className="scrollbar-thin absolute left-0 right-0 top-8 z-30 max-h-72 overflow-y-auto rounded-md border border-hairline bg-surface shadow-xl">
          {results.map((node) => (
            <button
              key={node.id}
              type="button"
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => {
                void focusNode(node.id)
                setOpen(false)
              }}
              className="flex w-full items-baseline gap-1.5 px-2 py-1.5 text-left transition-colors hover:bg-raised"
            >
              <span className="shrink-0 font-mono text-[0.58rem] uppercase text-faint">
                {node.kind.slice(0, 3)}
              </span>
              <span className="truncate font-mono text-2xs text-ink">{node.name}</span>
              <span className="ml-auto truncate font-mono text-[0.58rem] text-faint">{node.key}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export function GraphPane() {
  const entry = useActiveRepository()
  const nodes = useGraphStore((s) => s.nodes)
  const edges = useGraphStore((s) => s.edges)
  const totalNodes = useGraphStore((s) => s.totalNodes)
  const truncated = useGraphStore((s) => s.truncated)
  const loading = useGraphStore((s) => s.loading)
  const error = useGraphStore((s) => s.error)
  const highlight = useGraphStore((s) => s.highlight)
  const selectedId = useGraphStore((s) => s.selectedId)
  const clearEmphasis = useGraphStore((s) => s.clearEmphasis)

  if (!entry) return null
  if (!entry.status.graph_ready) return <BuildProgress entry={entry} />

  const files = nodes.filter((n) => n.kind === 'file').length
  const relationships = edges.filter((e) => e.kind !== 'CONTAINS').length
  const { status } = entry

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-hairline px-3 py-2">
        {nodes.length > 0 && (
          <>
            <Tag>{formatCount(files)} files</Tag>
            <Tag>{formatCount(nodes.length - files)} symbols</Tag>
            <Tag>{formatCount(relationships)} relationships</Tag>
            {truncated && (
              <Tag tone="accent" className="hidden lg:inline-flex">
                {formatCount(nodes.length)} most connected of {formatCount(totalNodes)}
              </Tag>
            )}
          </>
        )}

        {status.state === 'indexing_code' && (
          <span className="flex items-center gap-1.5 text-2xs text-faint">
            <StatusDot tone="accent" pulse />
            indexing code
            {status.progress_total > 0 &&
              ` ${Math.round((status.progress_done / status.progress_total) * 100)}%`}
          </span>
        )}
        {loading && (
          <span className="flex items-center gap-1.5 text-2xs text-faint">
            <StatusDot tone="accent" pulse />
            loading graph
          </span>
        )}

        <div className="ml-auto flex items-center gap-2">
          {(highlight || selectedId) && (
            <Button size="sm" onClick={clearEmphasis} title="Show the whole graph again">
              Clear highlight
            </Button>
          )}
          <GraphSearch />
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-1.5 border-b border-danger/30 bg-danger/[0.06] px-3 py-1.5 text-2xs text-danger">
          <IconWarn width={12} height={12} />
          {error}
        </div>
      )}

      <div className="min-h-0 flex-1">
        <KnowledgeGraph />
      </div>
    </div>
  )
}
