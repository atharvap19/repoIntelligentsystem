import { memo, useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import { CopyButton, Tag, Thinking } from '@/components/common/Bits'
import { IconFile, IconLayers, IconWarn } from '@/components/common/Icons'
import { cx, fileName, formatMs, toRelevance } from '@/lib/format'
import { chunkKey, useAppStore } from '@/store/useAppStore'
import type { NavigationTarget } from '@/lib/graphTypes'
import type { Message as MessageType, Source } from '@/lib/types'
import { INTENT_LABEL } from '@/components/graph/AIChat'

/**
 * The optional [Explore this →].
 *
 * Optional is the point (Part 5). The backend decides whether a meaningful
 * visual target exists and this renders nothing when it does not — "how many
 * Python files are there?" is answered, not explored, and a button under every
 * answer teaches people to ignore the button.
 *
 * The destination comes from structured data, never from parsing the prose
 * above it.
 */
function ExploreAction({ target }: { target: NavigationTarget }) {
  const exploreFrom = useAppStore((s) => s.exploreFrom)

  return (
    <button
      type="button"
      onClick={() => void exploreFrom(target)}
      className={cx(
        'mt-3 inline-flex items-center gap-2 rounded-lg border border-accent/45 bg-accent/[0.07]',
        'px-3 py-1.5 text-2xs font-medium text-accent transition-colors hover:bg-accent/12',
      )}
      title={`Open ${target.file || target.symbol || target.repository} in Explorer`}
    >
      <IconLayers width={12} height={12} />
      {target.label}
      <span aria-hidden>→</span>
    </button>
  )
}

/**
 * Citation chip.
 *
 * The system prompt asks the model to "mention filenames", and models reliably
 * put those in backticks — so inline code that matches a retrieved source is
 * upgraded into a live cross-link into the Inspector. Everything else renders
 * as ordinary inline code.
 */
function Citation({ source, index }: { source: Source; index: number }) {
  const focusChunk = useAppStore((s) => s.focusChunk)
  const focused = useAppStore((s) => s.focusedChunkKey === chunkKey(source))
  const openInspector = useAppStore((s) => s.inspectorOpen)
  const toggleInspector = useAppStore((s) => s.toggleInspector)

  return (
    <button
      type="button"
      onClick={() => {
        focusChunk(focused ? null : chunkKey(source))
        if (!openInspector) toggleInspector()
      }}
      onMouseEnter={() => focusChunk(chunkKey(source))}
      title={`Evidence ${index + 1} · ${source.file}`}
      className={cx(
        'mx-px inline-flex translate-y-px items-baseline gap-1 rounded border px-1.5 py-px align-baseline',
        'font-mono text-[0.8125em] transition-colors',
        focused
          ? 'border-accent bg-accent/15 text-accent'
          : 'border-hairline bg-raised text-ink hover:border-accent/50 hover:text-accent',
      )}
    >
      {fileName(source.file)}
      <sup className="text-[0.65em] font-semibold text-accent">{index + 1}</sup>
    </button>
  )
}

function SourceStrip({ sources }: { sources: Source[] }) {
  const focusChunk = useAppStore((s) => s.focusChunk)
  const focusedKey = useAppStore((s) => s.focusedChunkKey)
  const inspectorOpen = useAppStore((s) => s.inspectorOpen)
  const toggleInspector = useAppStore((s) => s.toggleInspector)

  return (
    <div className="mt-3.5 border-t border-hairline pt-2.5">
      <div className="mb-1.5 text-2xs uppercase tracking-[0.14em] text-faint">
        Grounded in {sources.length} chunk{sources.length === 1 ? '' : 's'}
      </div>
      <div className="flex flex-wrap gap-1.5">
        {sources.map((source, i) => {
          const key = chunkKey(source)
          const active = focusedKey === key
          return (
            <button
              key={key}
              type="button"
              onMouseEnter={() => focusChunk(key)}
              onMouseLeave={() => focusChunk(null)}
              onClick={() => {
                focusChunk(key)
                if (!inspectorOpen) toggleInspector()
              }}
              className={cx(
                'inline-flex max-w-full items-center gap-1.5 rounded-md border px-2 py-1 transition-colors',
                active
                  ? 'border-accent/60 bg-accent/10'
                  : 'border-hairline bg-surface hover:border-accent/40',
              )}
            >
              <span className="font-mono text-2xs font-semibold text-accent">{i + 1}</span>
              <IconFile width={11} height={11} className="shrink-0 text-faint" />
              <span className="truncate font-mono text-2xs text-muted">{source.file}</span>
              <span className="shrink-0 font-mono text-2xs text-faint">
                {(toRelevance(source.distance) * 100).toFixed(0)}
              </span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

function AssistantBody({ message }: { message: MessageType }) {
  // Index sources by both full path and bare filename so either spelling in the
  // model's prose resolves to the same chunk.
  const sourceIndex = useMemo(() => {
    const map = new Map<string, { source: Source; index: number }>()
    message.sources?.forEach((source, index) => {
      map.set(source.file.toLowerCase(), { source, index })
      map.set(fileName(source.file).toLowerCase(), { source, index })
    })
    return map
  }, [message.sources])

  if (message.pending) {
    return (
      <div className="flex items-center gap-2.5 py-1 text-sm text-muted">
        <Thinking />
        <span>Retrieving and generating…</span>
      </div>
    )
  }

  if (message.error) {
    return (
      <div className="flex items-start gap-2 rounded-lg border border-danger/35 bg-danger/[0.07] px-3 py-2.5">
        <IconWarn className="mt-0.5 shrink-0 text-danger" />
        <div className="text-sm leading-relaxed text-ink">
          <p className="font-medium text-danger">Request failed</p>
          <p className="mt-0.5 text-muted">{message.error}</p>
        </div>
      </div>
    )
  }

  return (
    <>
      <div className="answer">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeHighlight]}
          components={{
            code({ className, children, ...props }) {
              const text = String(children)
              const isBlock = /language-/.test(className ?? '')
              if (!isBlock) {
                const hit = sourceIndex.get(text.trim().toLowerCase())
                if (hit) return <Citation source={hit.source} index={hit.index} />
              }
              return (
                <code className={className} {...props}>
                  {children}
                </code>
              )
            },
          }}
        >
          {message.content}
        </ReactMarkdown>
      </div>

      {message.navigation?.available && <ExploreAction target={message.navigation} />}

      {message.sources && message.sources.length > 0 && <SourceStrip sources={message.sources} />}

      <div className="mt-2.5 flex flex-wrap items-center gap-2">
        <CopyButton value={message.content} label="Copy answer" />
        {message.intent && <Tag tone="accent">{INTENT_LABEL[message.intent] ?? message.intent}</Tag>}
        {message.focusLabel && <Tag tone="signal">{message.focusLabel}</Tag>}
        {/* How much of each evidence type reached the prompt — the honest
            version of "grounded in your code". */}
        {message.contextStats?.graph_nodes ? (
          <Tag>{message.contextStats.graph_nodes} graph nodes</Tag>
        ) : null}
        {message.latencyMs != null && <Tag>{formatMs(message.latencyMs)}</Tag>}
        {message.timings?.generate != null && <Tag>gen {formatMs(message.timings.generate)}</Tag>}
      </div>
    </>
  )
}

export const Message = memo(function Message({ message }: { message: MessageType }) {
  if (message.role === 'user') {
    return (
      <div className="animate-fade-up px-5 pt-6">
        <div className="mx-auto flex max-w-3xl gap-3">
          <div className="mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded border border-hairline bg-raised text-2xs font-semibold text-muted">
            You
          </div>
          <p className="whitespace-pre-wrap pt-0.5 text-[0.9375rem] font-medium leading-relaxed text-ink">
            {message.content}
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="animate-fade-up px-5 pt-3">
      <div className="mx-auto flex max-w-3xl gap-3">
        <div className="mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded bg-accent text-2xs font-bold text-canvas">
          R
        </div>
        <div className="min-w-0 flex-1">
          <AssistantBody message={message} />
        </div>
      </div>
    </div>
  )
})
