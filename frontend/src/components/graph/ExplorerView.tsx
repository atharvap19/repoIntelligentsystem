import { useEffect, useRef, useState } from 'react'
import { Button } from '@/components/common/Button'
import { StatusDot, Tag } from '@/components/common/Bits'
import {
  IconChevron,
  IconClose,
  IconLayers,
  IconSearch,
  IconWarn,
} from '@/components/common/Icons'
import { cx, formatCount } from '@/lib/format'
import * as graphApi from '@/lib/graphApi'
import type { GraphNode } from '@/lib/graphTypes'
import { useGraphStore } from '@/store/useGraphStore'
import { AIChat } from './AIChat'
import { CodeCells } from './CodeCell'
import { FileDetails } from './FileDetails'
import { RepositoryGraph } from './RepositoryGraph'
import { Timeline } from './Timeline'

function Breadcrumbs() {
  const path = useGraphStore((s) => s.path)
  const navigateTo = useGraphStore((s) => s.navigateTo)
  const truncated = useGraphStore((s) => s.truncated)
  const totalChildren = useGraphStore((s) => s.totalChildren)
  const nodes = useGraphStore((s) => s.nodes)

  return (
    <div className="flex items-center gap-1 overflow-x-auto px-3 py-1.5">
      {path.map((node, index) => (
        <span key={node.id} className="flex shrink-0 items-center gap-1">
          {index > 0 && <IconChevron width={11} height={11} className="text-faint" />}
          <button
            type="button"
            onClick={() => void navigateTo(index)}
            className={cx(
              'rounded px-1.5 py-0.5 font-mono text-2xs transition-colors',
              index === path.length - 1
                ? 'bg-raised text-ink'
                : 'text-muted hover:bg-raised hover:text-ink',
            )}
          >
            {node.name}
          </button>
        </span>
      ))}
      {truncated && (
        <Tag tone="accent" className="ml-2 shrink-0">
          showing {nodes.length} of {formatCount(totalChildren)}
        </Tag>
      )}
    </div>
  )
}

function GraphSearch() {
  const repository = useGraphStore((s) => s.repository)
  const focusById = useGraphStore((s) => s.focusById)
  const openFile = useGraphStore((s) => s.openFile)
  const [term, setTerm] = useState('')
  const [results, setResults] = useState<GraphNode[]>([])
  const [open, setOpen] = useState(false)
  const timer = useRef<number | null>(null)

  useEffect(() => {
    if (!repository || term.trim().length < 2) {
      setResults([])
      return
    }
    // Debounced: the graph search hits SQLite on every keystroke otherwise.
    if (timer.current) window.clearTimeout(timer.current)
    timer.current = window.setTimeout(async () => {
      try {
        const { results } = await graphApi.searchGraph(repository, term.trim(), 12)
        setResults(results)
        setOpen(true)
      } catch {
        setResults([])
      }
    }, 220)
    return () => {
      if (timer.current) window.clearTimeout(timer.current)
    }
  }, [term, repository])

  return (
    <div className="relative w-64">
      <IconSearch
        width={13}
        height={13}
        className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 text-faint"
      />
      <input
        value={term}
        onChange={(e) => setTerm(e.target.value)}
        onFocus={() => results.length && setOpen(true)}
        onBlur={() => window.setTimeout(() => setOpen(false), 150)}
        placeholder="Find a file or symbol…"
        spellCheck={false}
        className="h-7 w-full rounded-md border border-hairline bg-canvas pl-7 pr-2 font-mono text-2xs text-ink placeholder:text-faint focus:border-accent/60"
      />
      {open && results.length > 0 && (
        <div className="absolute left-0 right-0 top-8 z-30 max-h-72 overflow-y-auto rounded-md border border-hairline bg-surface shadow-xl scrollbar-thin">
          {results.map((node) => (
            <button
              key={node.id}
              type="button"
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => {
                void focusById(node.id)
                if (node.kind === 'file') void openFile(node.id, node.name, node.key)
                setOpen(false)
              }}
              className="flex w-full items-baseline gap-1.5 px-2 py-1.5 text-left transition-colors hover:bg-raised"
            >
              <span className="shrink-0 font-mono text-[0.58rem] uppercase text-faint">
                {node.kind.slice(0, 3)}
              </span>
              <span className="truncate font-mono text-2xs text-ink">{node.name}</span>
              <span className="ml-auto truncate font-mono text-[0.58rem] text-faint">
                {node.key}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

function RepositoryHeader() {
  const available = useGraphStore((s) => s.available)
  const repository = useGraphStore((s) => s.repository)
  const counts = useGraphStore((s) => s.counts)
  const selectRepository = useGraphStore((s) => s.selectRepository)
  const loading = useGraphStore((s) => s.loading)

  return (
    <div className="flex shrink-0 items-center gap-2 border-b border-hairline px-3 py-2">
      <IconLayers width={14} height={14} className="text-accent" />
      <select
        value={repository ?? ''}
        onChange={(e) => void selectRepository(e.target.value)}
        className="h-7 rounded-md border border-hairline bg-canvas px-2 font-mono text-xs text-ink focus:border-accent/60"
      >
        {available.map((entry) => (
          <option key={entry.repository} value={entry.repository}>
            {entry.repository}
          </option>
        ))}
      </select>

      {counts.file != null && (
        <div className="flex items-center gap-1.5">
          <Tag>{formatCount(counts.file)} files</Tag>
          <Tag>{formatCount((counts.class ?? 0) + (counts.function ?? 0) + (counts.method ?? 0))} symbols</Tag>
          <Tag>{formatCount(counts.edges ?? 0)} edges</Tag>
        </div>
      )}

      {loading && (
        <span className="flex items-center gap-1.5 text-2xs text-faint">
          <StatusDot tone="accent" pulse />
          loading
        </span>
      )}

      <div className="ml-auto">
        <GraphSearch />
      </div>
    </div>
  )
}

export function ExplorerView() {
  const bootstrap = useGraphStore((s) => s.bootstrap)
  const error = useGraphStore((s) => s.error)
  const available = useGraphStore((s) => s.available)
  const repository = useGraphStore((s) => s.repository)
  const [panelOpen, setPanelOpen] = useState(true)
  const [panelTab, setPanelTab] = useState<'details' | 'ask'>('details')
  const asking = useGraphStore((s) => s.asking)

  useEffect(() => {
    void bootstrap()
  }, [bootstrap])

  if (!repository && available.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center p-8">
        <div className="max-w-md text-center">
          <IconLayers width={26} height={26} className="mx-auto text-faint" />
          <h2 className="mt-3 text-sm font-semibold text-ink">No repository graph yet</h2>
          <p className="mt-1.5 text-xs leading-relaxed text-muted">
            Import a repository and index it — the graph is built as part of indexing. If the
            backend is not running, start it on port 8000.
          </p>
          {error && (
            <p className="mt-3 flex items-center justify-center gap-1.5 text-2xs text-danger">
              <IconWarn width={12} height={12} />
              {error}
            </p>
          )}
          <Button size="sm" variant="outline" className="mt-4" onClick={() => void bootstrap()}>
            Retry
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="flex min-h-0 flex-1">
      <section className="flex min-w-0 flex-1 flex-col">
        <RepositoryHeader />
        <Timeline />
        <Breadcrumbs />

        {error && (
          <div className="flex items-center gap-1.5 border-b border-danger/30 bg-danger/[0.06] px-3 py-1.5 text-2xs text-danger">
            <IconWarn width={12} height={12} />
            {error}
          </div>
        )}

        {/* The graph keeps its own space; code cells stack beneath it and
            never replace it. */}
        <div className="min-h-0 flex-1">
          <RepositoryGraph />
        </div>

        <CodeCells />
      </section>

      <aside
        className={cx(
          'flex shrink-0 flex-col border-l border-hairline bg-surface/60 transition-[width]',
          panelOpen ? 'w-[340px]' : 'w-9',
        )}
      >
        <div className="flex h-9 shrink-0 items-center gap-0.5 border-b border-hairline px-1.5">
          <Button
            size="icon"
            onClick={() => setPanelOpen((v) => !v)}
            title={panelOpen ? 'Collapse panel' : 'Expand panel'}
            aria-label="Toggle details panel"
          >
            {panelOpen ? <IconClose width={13} height={13} /> : <IconChevron width={13} height={13} className="rotate-180" />}
          </Button>
          {panelOpen &&
            (['details', 'ask'] as const).map((id) => (
              <button
                key={id}
                type="button"
                onClick={() => setPanelTab(id)}
                className={cx(
                  'flex h-6 items-center gap-1 rounded px-2 text-2xs transition-colors',
                  panelTab === id ? 'bg-raised text-ink' : 'text-muted hover:text-ink',
                )}
              >
                {id === 'details' ? 'Details' : 'Ask AI'}
                {id === 'ask' && asking && <StatusDot tone="accent" pulse />}
              </button>
            ))}
        </div>
        {panelOpen &&
          (panelTab === 'details' ? (
            <div className="scrollbar-thin flex-1 overflow-y-auto">
              <FileDetails />
            </div>
          ) : (
            <AIChat />
          ))}
      </aside>
    </div>
  )
}
