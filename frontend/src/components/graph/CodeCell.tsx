import { useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import { Button } from '@/components/common/Button'
import { CopyButton, Tag } from '@/components/common/Bits'
import { IconChevron, IconClose, IconFile, IconWarn } from '@/components/common/Icons'
import { cx, languageOf } from '@/lib/format'
import { useGraphStore, type OpenFile } from '@/store/useGraphStore'

/**
 * Source view for one file, rendered *below* the graph rather than replacing
 * it — keeping the graph on screen while reading code is the core of the
 * interaction model.
 */
function Cell({ file }: { file: OpenFile }) {
  const close = useGraphStore((s) => s.closeFile)
  const toggle = useGraphStore((s) => s.toggleFileCollapsed)
  const [jumpTo, setJumpTo] = useState<number | null>(null)
  const bodyRef = useRef<HTMLDivElement>(null)

  const language = languageOf(file.path, file.content?.language)

  // Line numbers are rendered as their own column rather than baked into the
  // code text, so copying yields source without numbering.
  const lines = useMemo(() => (file.content?.content ?? '').split('\n'), [file.content])

  function scrollToLine(line: number) {
    setJumpTo(line)
    const container = bodyRef.current
    const target = container?.querySelector<HTMLElement>(`[data-line="${line}"]`)
    if (target && container) {
      container.scrollTo({ top: target.offsetTop - 48, behavior: 'smooth' })
    }
  }

  return (
    <div className="shrink-0 overflow-hidden rounded-lg border border-hairline bg-surface">
      <div className="flex items-center gap-2 border-b border-hairline px-2.5 py-1.5">
        <Button
          size="icon"
          onClick={() => toggle(file.nodeId)}
          title={file.collapsed ? 'Expand' : 'Collapse'}
          aria-label={file.collapsed ? 'Expand' : 'Collapse'}
        >
          <IconChevron
            width={13}
            height={13}
            className={cx('transition-transform', !file.collapsed && 'rotate-90')}
          />
        </Button>

        <IconFile width={12} height={12} className="shrink-0 text-faint" />
        <span className="truncate font-mono text-xs font-medium text-ink">{file.name}</span>
        <span className="truncate font-mono text-2xs text-faint">{file.path}</span>

        <div className="ml-auto flex shrink-0 items-center gap-1.5">
          {file.content && (
            <>
              <Tag>{language}</Tag>
              <Tag>{file.content.line_count} lines</Tag>
              {file.content.redacted && (
                <Tag tone="danger" className="cursor-help">
                  redacted
                </Tag>
              )}
              {file.content.truncated && <Tag tone="accent">truncated</Tag>}
              <CopyButton value={file.content.content} label="Copy file" />
            </>
          )}
          <Button
            size="icon"
            variant="danger"
            onClick={() => close(file.nodeId)}
            title="Close"
            aria-label={`Close ${file.name}`}
          >
            <IconClose width={13} height={13} />
          </Button>
        </div>
      </div>

      {!file.collapsed && (
        <>
          {file.content && file.content.symbols.length > 0 && (
            <div className="flex flex-wrap gap-1 border-b border-hairline px-2.5 py-1.5">
              <span className="text-2xs uppercase tracking-wide text-faint">jump to</span>
              {file.content.symbols.slice(0, 24).map((symbol) => (
                <button
                  key={symbol.id}
                  type="button"
                  onClick={() => scrollToLine(symbol.data.start_line ?? 1)}
                  className={cx(
                    'rounded border px-1.5 py-px font-mono text-2xs transition-colors',
                    jumpTo === symbol.data.start_line
                      ? 'border-accent bg-accent/10 text-accent'
                      : 'border-hairline text-muted hover:border-accent/50 hover:text-ink',
                  )}
                  title={`${symbol.kind} · lines ${symbol.data.start_line}-${symbol.data.end_line}`}
                >
                  {symbol.name}
                </button>
              ))}
            </div>
          )}

          <div ref={bodyRef} className="scrollbar-thin max-h-[46vh] overflow-auto">
            {file.loading && (
              <div className="px-3 py-6 text-center text-xs text-faint">Loading source…</div>
            )}

            {file.error && (
              <div className="flex items-start gap-2 px-3 py-4 text-xs text-danger">
                <IconWarn width={14} height={14} className="mt-px shrink-0" />
                {file.error}
              </div>
            )}

            {file.content && (
              <div className="flex font-mono text-[0.78rem] leading-[1.55]">
                <div
                  aria-hidden
                  className="select-none border-r border-hairline bg-raised px-2 py-2 text-right text-faint"
                >
                  {lines.map((_, index) => (
                    <div key={index} data-line={index + 1}>
                      {index + 1}
                    </div>
                  ))}
                </div>
                <div className="answer min-w-0 flex-1 overflow-x-auto py-2 [&_pre]:!m-0 [&_pre]:!border-0 [&_pre]:!bg-transparent [&_pre]:!p-0 [&_pre]:!px-3">
                  <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
                    {`\`\`\`${language}\n${file.content.content}\n\`\`\``}
                  </ReactMarkdown>
                </div>
              </div>
            )}
          </div>

          {file.content?.redacted && (
            <div className="border-t border-hairline px-2.5 py-1.5 text-2xs text-faint">
              Possible credentials were masked before this file left the backend:{' '}
              {file.content.redaction_findings.join(', ')}
            </div>
          )}
        </>
      )}
    </div>
  )
}

export function CodeCells() {
  const openFiles = useGraphStore((s) => s.openFiles)
  if (openFiles.length === 0) return null

  return (
    <div className="scrollbar-thin flex max-h-[52vh] shrink-0 flex-col gap-2 overflow-y-auto border-t border-hairline bg-canvas p-2">
      {openFiles.map((file) => (
        <Cell key={file.nodeId} file={file} />
      ))}
    </div>
  )
}
