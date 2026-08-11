import { useMemo, useState } from 'react'
import { EmptyHint, Tag } from '@/components/common/Bits'
import { IconLayers, IconSliders } from '@/components/common/Icons'
import { cx, formatMs, languageColor, languageOf, toRelevance } from '@/lib/format'
import { useAppStore, useLatestSources } from '@/store/useAppStore'
import type { Source } from '@/lib/types'
import { ChunkCard } from './ChunkCard'
import { RunConfig } from './RunConfig'

/** embed → search → generate, drawn as proportional segments. */
function PipelineTrace({
  timings,
  total,
}: {
  timings?: Partial<Record<'embed' | 'search' | 'generate', number>>
  total?: number
}) {
  const stages = [
    { key: 'embed' as const, label: 'embed', color: 'bg-signal' },
    { key: 'search' as const, label: 'search', color: 'bg-signal/60' },
    { key: 'generate' as const, label: 'generate', color: 'bg-accent' },
  ]

  const measured = stages.filter((s) => timings?.[s.key] != null)
  const sum = measured.reduce((acc, s) => acc + (timings?.[s.key] ?? 0), 0)

  if (measured.length === 0) {
    return (
      <div className="px-3 py-2.5">
        <div className="flex items-baseline justify-between">
          <span className="text-2xs uppercase tracking-[0.14em] text-faint">Round trip</span>
          <span className="font-mono text-xs text-ink">{formatMs(total)}</span>
        </div>
        <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-hairline">
          <div className="h-full w-full bg-gradient-to-r from-signal to-accent opacity-60" />
        </div>
        <p className="mt-1.5 text-2xs leading-relaxed text-faint">
          Per-stage timings appear once the backend returns a{' '}
          <code className="font-mono">timings</code> object.
        </p>
      </div>
    )
  }

  return (
    <div className="px-3 py-2.5">
      <div className="flex items-baseline justify-between">
        <span className="text-2xs uppercase tracking-[0.14em] text-faint">Pipeline</span>
        <span className="font-mono text-xs text-ink">{formatMs(total ?? sum)}</span>
      </div>
      <div className="mt-1.5 flex h-1 gap-px overflow-hidden rounded-full bg-hairline">
        {measured.map((s) => (
          <div
            key={s.key}
            className={s.color}
            style={{ width: `${((timings?.[s.key] ?? 0) / sum) * 100}%` }}
            title={`${s.label} ${formatMs(timings?.[s.key])}`}
          />
        ))}
      </div>
      <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1">
        {measured.map((s) => (
          <span key={s.key} className="flex items-center gap-1.5 text-2xs text-muted">
            <span className={cx('h-1.5 w-1.5 rounded-full', s.color)} />
            {s.label}
            <span className="font-mono text-faint">{formatMs(timings?.[s.key])}</span>
          </span>
        ))}
      </div>
    </div>
  )
}

/** How concentrated the evidence is — many chunks from one file vs. spread out. */
function CoverageSummary({ sources }: { sources: Source[] }) {
  const files = useMemo(() => {
    const map = new Map<string, number>()
    sources.forEach((s) => map.set(s.file, (map.get(s.file) ?? 0) + 1))
    return [...map.entries()].sort((a, b) => b[1] - a[1])
  }, [sources])

  const best = toRelevance(Math.min(...sources.map((s) => s.distance)))
  const worst = toRelevance(Math.max(...sources.map((s) => s.distance)))

  return (
    <div className="space-y-2 px-3 py-2.5">
      <div className="flex flex-wrap gap-1.5">
        <Tag tone="accent">{sources.length} chunks</Tag>
        <Tag>{files.length} files</Tag>
        <Tag>
          {(best * 100).toFixed(0)}–{(worst * 100).toFixed(0)} rel
        </Tag>
      </div>

      <div className="space-y-1">
        {files.slice(0, 4).map(([file, count]) => {
          const language = languageOf(file)
          return (
            <div key={file} className="flex items-center gap-1.5">
              <span
                className="h-1.5 w-1.5 shrink-0 rounded-full"
                style={{ background: languageColor(language) }}
              />
              <span className="truncate font-mono text-2xs text-muted">{file}</span>
              {count > 1 && <span className="shrink-0 font-mono text-2xs text-accent">×{count}</span>}
            </div>
          )
        })}
        {files.length > 4 && (
          <div className="pl-3 text-2xs text-faint">+{files.length - 4} more</div>
        )}
      </div>
    </div>
  )
}

function EvidenceTab() {
  const { sources } = useLatestSources()
  const asking = useAppStore((s) => s.asking)
  const minRelevance = useAppStore((s) => s.settings.minRelevance)
  const messages = useAppStore((s) => s.messages)

  const latency = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i]
      if (m.role === 'assistant' && !m.pending) return { ms: m.latencyMs, timings: m.timings }
    }
    return null
  }, [messages])

  if (asking && sources.length === 0) {
    return (
      <div className="space-y-2 p-3">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="relative h-16 overflow-hidden rounded-lg border border-hairline bg-surface"
          >
            <div className="absolute inset-0 -translate-x-full animate-shimmer bg-gradient-to-r from-transparent via-raised to-transparent" />
          </div>
        ))}
      </div>
    )
  }

  if (sources.length === 0) {
    return (
      <EmptyHint icon={<IconLayers width={22} height={22} />} title="No retrieval yet">
        Ask a question and every chunk pulled from the vector store lands here — ranked, scored, and
        readable in full.
      </EmptyHint>
    )
  }

  const visible = sources.filter((s) => toRelevance(s.distance) >= minRelevance)
  const hidden = sources.length - visible.length

  return (
    <div>
      <div className="divide-y divide-hairline border-b border-hairline">
        <PipelineTrace timings={latency?.timings} total={latency?.ms} />
        <CoverageSummary sources={sources} />
      </div>

      <div className="space-y-2 p-3">
        {visible.map((source, i) => (
          <ChunkCard key={`${source.file}#${source.chunk_index ?? i}`} source={source} rank={i} />
        ))}

        {hidden > 0 && (
          <p className="px-1 pt-1 text-2xs text-faint">
            {hidden} chunk{hidden === 1 ? '' : 's'} hidden by the relevance filter — the model still
            received {sources.length === 1 ? 'it' : 'them'}.
          </p>
        )}
      </div>
    </div>
  )
}

export function RetrievalInspector() {
  const [tab, setTab] = useState<'evidence' | 'config'>('evidence')
  const { sources } = useLatestSources()

  const tabs = [
    { id: 'evidence' as const, label: 'Evidence', icon: IconLayers, badge: sources.length || null },
    { id: 'config' as const, label: 'Run config', icon: IconSliders, badge: null },
  ]

  return (
    <aside className="flex w-[356px] shrink-0 flex-col border-l border-hairline bg-surface/60">
      <div className="flex h-10 shrink-0 items-center gap-1 border-b border-hairline px-2">
        {tabs.map(({ id, label, icon: Icon, badge }) => (
          <button
            key={id}
            type="button"
            onClick={() => setTab(id)}
            className={cx(
              'flex h-7 items-center gap-1.5 rounded-md px-2.5 text-xs transition-colors',
              tab === id ? 'bg-raised text-ink' : 'text-muted hover:text-ink',
            )}
          >
            <Icon width={13} height={13} className={cx(tab === id && 'text-accent')} />
            {label}
            {badge != null && (
              <span className="rounded bg-accent/15 px-1 font-mono text-2xs text-accent">
                {badge}
              </span>
            )}
          </button>
        ))}
      </div>

      <div className="scrollbar-thin flex-1 overflow-y-auto">
        {tab === 'evidence' ? <EvidenceTab /> : <RunConfig />}
      </div>

      <div className="shrink-0 border-t border-hairline px-3 py-2 text-2xs leading-relaxed text-faint">
        Relevance is <span className="font-mono">1/(1+distance)</span> — a display rescaling of
        Chroma's raw distance, not a probability.
      </div>
    </aside>
  )
}
