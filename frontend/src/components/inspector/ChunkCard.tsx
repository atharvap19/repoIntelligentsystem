import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import { CopyButton, ScoreBar, Tag } from '@/components/common/Bits'
import { IconChevron } from '@/components/common/Icons'
import {
  cx,
  dirName,
  fileName,
  languageColor,
  languageOf,
  relevanceLabel,
  toRelevance,
} from '@/lib/format'
import { chunkKey, useAppStore } from '@/store/useAppStore'
import type { Source } from '@/lib/types'

export function ChunkCard({ source, rank }: { source: Source; rank: number }) {
  const key = chunkKey(source)
  const focused = useAppStore((s) => s.focusedChunkKey === key)
  const focusChunk = useAppStore((s) => s.focusChunk)
  const [open, setOpen] = useState(rank === 0)
  const ref = useRef<HTMLDivElement>(null)

  // Scroll into view when a citation in the answer selects this chunk.
  useEffect(() => {
    if (focused) ref.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [focused])

  const relevance = toRelevance(source.distance)
  const language = languageOf(source.file, source.language)
  const directory = dirName(source.file)

  return (
    <div
      ref={ref}
      onMouseEnter={() => focusChunk(key)}
      onMouseLeave={() => focusChunk(null)}
      className={cx(
        'rounded-lg border transition-colors',
        focused ? 'border-accent/60 bg-accent/[0.06]' : 'border-hairline bg-surface',
      )}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-start gap-2.5 p-2.5 text-left"
        aria-expanded={open}
      >
        <span
          className={cx(
            'mt-px grid h-5 w-5 shrink-0 place-items-center rounded font-mono text-2xs font-bold',
            rank === 0 ? 'bg-accent text-canvas' : 'bg-raised text-muted',
          )}
        >
          {rank + 1}
        </span>

        <span className="min-w-0 flex-1">
          <span className="flex items-baseline gap-1.5">
            <span
              className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full"
              style={{ background: languageColor(language) }}
              aria-hidden
            />
            <span className="truncate font-mono text-xs font-medium text-ink">
              {fileName(source.file)}
            </span>
            {source.chunk_index != null && (
              <span className="shrink-0 font-mono text-2xs text-faint">#{source.chunk_index}</span>
            )}
          </span>

          {directory && (
            <span className="mt-0.5 block truncate font-mono text-2xs text-faint">{directory}/</span>
          )}

          <span className="mt-2 flex items-center gap-2">
            <ScoreBar value={relevance} className="flex-1" />
            <span className="shrink-0 font-mono text-2xs tabular-nums text-muted">
              {(relevance * 100).toFixed(0)}
            </span>
          </span>
        </span>

        <IconChevron
          width={13}
          height={13}
          className={cx('mt-1 shrink-0 text-faint transition-transform', open && 'rotate-90')}
        />
      </button>

      {open && (
        <div className="border-t border-hairline px-2.5 pb-2.5 pt-2">
          <div className="mb-2 flex flex-wrap items-center gap-1.5">
            <Tag tone={relevance >= 0.55 ? 'signal' : 'neutral'}>{relevanceLabel(source.distance)}</Tag>
            <Tag>d={source.distance.toFixed(4)}</Tag>
            <Tag>{language}</Tag>
            {source.content && <Tag>{source.content.length} ch</Tag>}
            {source.content && <CopyButton value={source.content} label="Copy chunk" />}
          </div>

          {source.content ? (
            <div className="answer max-h-72 overflow-auto scrollbar-thin">
              <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
                {`\`\`\`${language}\n${source.content}\n\`\`\``}
              </ReactMarkdown>
            </div>
          ) : (
            <p className="rounded border border-dashed border-hairline px-2.5 py-3 text-2xs leading-relaxed text-faint">
              The backend returns only <code className="font-mono">file</code> and{' '}
              <code className="font-mono">distance</code> for each source. Add{' '}
              <code className="font-mono">content</code> to the sources list in{' '}
              <code className="font-mono">chat_service.py</code> to show the retrieved text here.
            </p>
          )}
        </div>
      )}
    </div>
  )
}
