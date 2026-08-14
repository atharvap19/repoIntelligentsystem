import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import { Button } from '@/components/common/Button'
import { Tag, Thinking } from '@/components/common/Bits'
import { IconFile, IconRefresh, IconSend, IconWarn } from '@/components/common/Icons'
import { cx } from '@/lib/format'
import type { Message } from '@/lib/types'
import { useAppStore } from '@/store/useAppStore'
import { useGraphStore } from '@/store/useGraphStore'

/** Human-readable labels for the router's intents. */
export const INTENT_LABEL: Record<string, string> = {
  repository_search: 'search',
  code_explanation: 'explain',
  architecture_analysis: 'architecture',
  structure_analysis: 'structure',
  dependency_analysis: 'dependencies',
  file_analysis: 'file',
  git_history_analysis: 'history',
  flow_analysis: 'flow',
  concept_analysis: 'concept',
  hotspot_analysis: 'hotspots',
  readonly_request: 'read-only',
}

/** Questions that each exercise a different branch of the router. */
const STARTERS = [
  'What is this repository about?',
  'Explain the architecture.',
  'How does a request flow through this repository?',
  'Which files are most depended on?',
]

function Turn({ turn }: { turn: Message }) {
  const openFile = useGraphStore((s) => s.openFile)
  const applyNavigation = useGraphStore((s) => s.applyNavigation)

  if (turn.role === 'user') {
    return (
      <div className="px-3 pt-3">
        <p className="whitespace-pre-wrap text-[0.8125rem] font-medium leading-relaxed text-ink">
          {turn.content}
        </p>
      </div>
    )
  }

  if (turn.pending) {
    return (
      <div className="flex items-center gap-2 px-3 py-2 text-2xs text-muted">
        <Thinking />
        <span>Retrieving, reading the graph and generating — this runs locally.</span>
      </div>
    )
  }

  if (turn.error) {
    return (
      <div className="mx-3 my-2 flex items-start gap-1.5 rounded border border-danger/35 bg-danger/[0.07] px-2 py-1.5 text-2xs text-danger">
        <IconWarn width={12} height={12} className="mt-px shrink-0" />
        {turn.error}
      </div>
    )
  }

  return (
    <div className="px-3 pb-3 pt-1.5">
      <div className="mb-1.5 flex flex-wrap items-center gap-1">
        {turn.intent && <Tag tone="accent">{INTENT_LABEL[turn.intent] ?? turn.intent}</Tag>}
        {turn.confidence != null && <Tag>conf {turn.confidence.toFixed(2)}</Tag>}
        {turn.focusLabel && <Tag tone="signal">{turn.focusLabel}</Tag>}
        {turn.contextStats?.graph_nodes ? (
          <Tag>{turn.contextStats.graph_nodes} graph nodes</Tag>
        ) : null}
      </div>

      <div className="answer text-[0.8125rem] [&_pre]:text-[0.7rem]">
        <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
          {turn.content}
        </ReactMarkdown>
      </div>

      {/* Already in Explorer, so this moves the graph rather than switching
          views — the same target, one step shorter. */}
      {turn.navigation?.available && (
        <button
          type="button"
          onClick={() => void applyNavigation(turn.navigation!)}
          className="mt-2 w-full rounded border border-accent/40 bg-accent/[0.07] px-2 py-1.5 text-left text-2xs text-accent transition-colors hover:bg-accent/10"
        >
          {turn.navigation.label} →
        </button>
      )}

      {turn.sources && turn.sources.length > 0 && (
        <div className="mt-2 border-t border-hairline pt-1.5">
          <div className="mb-1 text-[0.58rem] uppercase tracking-wide text-faint">
            grounded in {turn.sources.length} chunks
          </div>
          <div className="space-y-0.5">
            {turn.sources.slice(0, 6).map((source, index) => (
              <button
                key={`${source.file}-${index}`}
                type="button"
                onClick={() => {
                  // Sources carry a path; the graph node id is derivable.
                  const repository = source.repository
                  if (repository && source.relative_path) {
                    void openFile(
                      `${repository}::file::${source.relative_path}`,
                      source.file_name ?? source.file,
                      source.relative_path,
                    )
                  }
                }}
                className="flex w-full items-center gap-1.5 rounded px-1 py-0.5 text-left transition-colors hover:bg-raised"
                title={source.file}
              >
                <IconFile width={10} height={10} className="shrink-0 text-faint" />
                <span className="truncate font-mono text-[0.62rem] text-muted">
                  {source.symbol || source.file}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

/**
 * The Explorer's view of the conversation.
 *
 * Not a second chat: it renders `useAppStore.messages`, the same array the
 * Chat view renders. Switching views changes the layout around the thread and
 * nothing else, which is what Part 4 asks for — and it is why asking here and
 * then switching to Chat shows the answer already there.
 */
export function AIChat() {
  const messages = useAppStore((s) => s.messages)
  const asking = useAppStore((s) => s.asking)
  const ask = useAppStore((s) => s.ask)
  const clear = useAppStore((s) => s.clearConversation)
  const repository = useGraphStore((s) => s.repository)
  const selected = useGraphStore((s) => s.selected)

  const [value, setValue] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  function submit() {
    const question = value.trim()
    if (!question || asking) return
    setValue('')
    void ask(question)
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex shrink-0 items-center gap-1.5 border-b border-hairline px-3 py-1.5">
        <span className="truncate font-mono text-2xs text-muted">{repository}</span>
        {messages.length > 0 && (
          <Button size="sm" className="ml-auto" onClick={clear} title="Clear conversation">
            <IconRefresh width={11} height={11} />
            New
          </Button>
        )}
      </div>

      <div className="scrollbar-thin min-h-0 flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <div className="p-3">
            <p className="text-2xs leading-relaxed text-faint">
              Answers combine vector search, keyword search and the repository graph. Selecting a
              node first lets follow-ups just say &ldquo;this&rdquo;. This is the same conversation
              as the Chat tab.
            </p>
            <div className="mt-2.5 space-y-1">
              {STARTERS.map((starter) => (
                <button
                  key={starter}
                  type="button"
                  onClick={() => void ask(starter)}
                  className="w-full rounded border border-hairline px-2 py-1.5 text-left text-2xs text-muted transition-colors hover:border-accent/45 hover:text-ink"
                >
                  {starter}
                </button>
              ))}
              {selected && (
                <button
                  type="button"
                  onClick={() => void ask('What depends on this?')}
                  className="w-full rounded border border-accent/40 bg-accent/[0.07] px-2 py-1.5 text-left text-2xs text-accent transition-colors hover:bg-accent/10"
                >
                  What depends on {selected.name}?
                </button>
              )}
            </div>
          </div>
        ) : (
          <>
            {messages.map((turn) => (
              <Turn key={turn.id} turn={turn} />
            ))}
            <div ref={bottomRef} />
          </>
        )}
      </div>

      <div className="shrink-0 border-t border-hairline p-2">
        {selected && (
          <div className="mb-1.5 flex items-center gap-1 px-0.5 text-[0.58rem] text-faint">
            <span className="uppercase tracking-wide">asking about</span>
            <span className="truncate font-mono text-accent">{selected.name}</span>
          </div>
        )}
        <div
          className={cx(
            'flex items-end gap-1.5 rounded-lg border bg-canvas px-2 py-1.5 transition-colors',
            'border-hairline focus-within:border-accent/55',
          )}
        >
          <textarea
            rows={2}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                submit()
              }
            }}
            placeholder={selected ? `Ask about ${selected.name}…` : 'Ask about this repository…'}
            className="max-h-28 flex-1 resize-none bg-transparent text-2xs leading-relaxed text-ink outline-none placeholder:text-faint"
          />
          <Button
            variant="primary"
            size="icon"
            onClick={submit}
            disabled={!value.trim() || asking}
            title="Ask (Enter)"
            aria-label="Ask"
          >
            <IconSend width={13} height={13} />
          </Button>
        </div>
      </div>
    </div>
  )
}
