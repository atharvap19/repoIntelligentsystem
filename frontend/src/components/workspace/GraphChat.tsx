import { useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import { Button } from '@/components/common/Button'
import { Tag, Thinking } from '@/components/common/Bits'
import { IconClose, IconFile, IconGraph, IconRefresh, IconSend, IconWarn } from '@/components/common/Icons'
import { cx } from '@/lib/format'
import type { Message } from '@/lib/types'
import { useChatStore } from '@/store/useChatStore'
import { useGraphStore, useSelectedNode } from '@/store/useGraphStore'
import { useActiveRepository } from '@/store/useWorkspaceStore'

/** Human-readable labels for the agent's intents. */
const INTENT_LABEL: Record<string, string> = {
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

const STARTERS = [
  'What is this repository about?',
  'Explain the architecture.',
  'Which files are most depended on?',
  'How does a request flow through this repository?',
]

function Turn({ turn }: { turn: Message }) {
  const showAnswer = useGraphStore((s) => s.showAnswer)
  const focusNode = useGraphStore((s) => s.focusNode)
  const repository = useGraphStore((s) => s.repository)
  const current = useGraphStore((s) => s.highlight)

  if (turn.role === 'user') {
    return (
      <div className="px-4 pt-4">
        <p className="whitespace-pre-wrap text-[0.8125rem] font-medium leading-relaxed text-ink">
          {turn.content}
        </p>
      </div>
    )
  }

  if (turn.pending) {
    return (
      <div className="flex items-center gap-2 px-4 py-3 text-2xs text-muted">
        <Thinking />
        <span>Reading the graph and generating — this runs locally and can take a few minutes.</span>
      </div>
    )
  }

  if (turn.error) {
    return (
      <div className="mx-4 my-2 flex items-start gap-1.5 rounded border border-danger/35 bg-danger/[0.07] px-2 py-1.5 text-2xs text-danger">
        <IconWarn width={12} height={12} className="mt-px shrink-0" />
        {turn.error}
      </div>
    )
  }

  const highlighted = turn.highlight?.node_ids.length ?? 0
  const onScreen = current !== null && current.node_ids[0] === turn.highlight?.node_ids[0]
  const files = [...new Map((turn.sources ?? []).map((s) => [s.relative_path || s.file, s])).values()]

  return (
    <div className="px-4 pb-4 pt-2">
      <div className="mb-1.5 flex flex-wrap items-center gap-1">
        {turn.intent && <Tag tone="accent">{INTENT_LABEL[turn.intent] ?? turn.intent}</Tag>}
        {turn.focusLabel && <Tag tone="signal">{turn.focusLabel}</Tag>}
      </div>

      <div className="answer text-[0.8125rem] [&_pre]:text-[0.7rem]">
        <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
          {turn.content}
        </ReactMarkdown>
      </div>

      {highlighted > 0 && (
        <button
          type="button"
          onClick={() => void showAnswer(turn.highlight!)}
          className={cx(
            'mt-2.5 flex w-full items-center gap-1.5 rounded border px-2 py-1.5 text-left text-2xs transition-colors',
            onScreen
              ? 'border-accent/40 bg-accent/[0.07] text-accent'
              : 'border-hairline text-muted hover:border-accent/40 hover:text-accent',
          )}
        >
          <IconGraph width={12} height={12} />
          {onScreen ? `${highlighted} nodes highlighted in the graph` : `Show the ${highlighted} nodes in the graph`}
        </button>
      )}

      {files.length > 0 && (
        <div className="mt-2 border-t border-hairline pt-1.5">
          <div className="mb-1 text-[0.58rem] uppercase tracking-wide text-faint">
            code it quoted
          </div>
          <div className="space-y-0.5">
            {files.slice(0, 6).map((source) => (
              <button
                key={source.relative_path || source.file}
                type="button"
                onClick={() =>
                  repository && void focusNode(`${repository}::file::${source.relative_path || source.file}`)
                }
                className="flex w-full items-center gap-1.5 rounded px-1 py-0.5 text-left transition-colors hover:bg-raised"
                title="Show in the graph"
              >
                <IconFile width={10} height={10} className="shrink-0 text-faint" />
                <span className="truncate font-mono text-[0.62rem] text-muted">
                  {source.relative_path || source.file}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export function GraphChat() {
  const entry = useActiveRepository()
  const messages = useChatStore((s) => s.messages)
  const asking = useChatStore((s) => s.asking)
  const draft = useChatStore((s) => s.draft)
  const setDraft = useChatStore((s) => s.setDraft)
  const ask = useChatStore((s) => s.ask)
  const clear = useChatStore((s) => s.clear)
  const graphLoaded = useGraphStore((s) => s.nodes.length > 0)
  const select = useGraphStore((s) => s.select)
  const selected = useSelectedNode()

  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  // A graph action that prefills the composer should also hand it the cursor.
  useEffect(() => {
    if (draft && document.activeElement !== inputRef.current) {
      inputRef.current?.focus()
      inputRef.current?.setSelectionRange(draft.length, draft.length)
    }
  }, [draft])

  const ready = !!entry?.status.graph_ready && graphLoaded
  const status = entry?.status

  function submit() {
    if (!draft.trim() || asking || !ready) return
    void ask(draft)
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex h-10 shrink-0 items-center gap-2 border-b border-hairline px-4">
        <span className="text-xs font-semibold text-ink">Talk to the graph</span>
        {messages.length > 0 && (
          <Button size="sm" className="ml-auto" onClick={clear} title="Start a new conversation">
            <IconRefresh width={11} height={11} />
            New
          </Button>
        )}
      </div>

      {status?.graph_ready && !status.search_ready && (
        <div
          className={cx(
            'flex shrink-0 items-start gap-1.5 border-b px-4 py-2 text-2xs leading-relaxed',
            status.state === 'error'
              ? 'border-danger/30 bg-danger/[0.06] text-danger'
              : 'border-hairline bg-raised/50 text-muted',
          )}
        >
          <IconWarn width={12} height={12} className="mt-0.5 shrink-0" />
          {status.state === 'error'
            ? status.message
            : 'Code is still being indexed. You can ask now — answers use the graph until code search is ready.'}
        </div>
      )}

      <div className="scrollbar-thin min-h-0 flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <div className="p-4">
            <p className="text-2xs leading-relaxed text-faint">
              Ask anything about {entry?.repository ?? 'the repository'}. Answers come from the
              knowledge graph and the code, and the nodes each answer uses light up on the graph.
              Click a node first to ask about it.
            </p>
            <div className="mt-3 space-y-1">
              {selected && (
                <>
                  <button
                    type="button"
                    disabled={!ready || asking}
                    onClick={() => void ask(`What does ${selected.name} do?`)}
                    className="w-full rounded border border-accent/40 bg-accent/[0.07] px-2 py-1.5 text-left text-2xs text-accent transition-colors hover:bg-accent/10 disabled:opacity-50"
                  >
                    What does {selected.name} do?
                  </button>
                  <button
                    type="button"
                    disabled={!ready || asking}
                    onClick={() => void ask(`What depends on ${selected.name}?`)}
                    className="w-full rounded border border-accent/40 bg-accent/[0.07] px-2 py-1.5 text-left text-2xs text-accent transition-colors hover:bg-accent/10 disabled:opacity-50"
                  >
                    What depends on {selected.name}?
                  </button>
                </>
              )}
              {STARTERS.map((starter) => (
                <button
                  key={starter}
                  type="button"
                  disabled={!ready || asking}
                  onClick={() => void ask(starter)}
                  className="w-full rounded border border-hairline px-2 py-1.5 text-left text-2xs text-muted transition-colors hover:border-accent/45 hover:text-ink disabled:opacity-50"
                >
                  {starter}
                </button>
              ))}
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

      <div className="shrink-0 border-t border-hairline p-3">
        {selected && (
          <div className="mb-1.5 flex items-center gap-1 px-0.5 text-[0.62rem] text-faint">
            <span className="uppercase tracking-wide">asking about</span>
            <span className="truncate font-mono text-accent">{selected.name}</span>
            <button
              type="button"
              onClick={() => select(null)}
              className="ml-0.5 rounded p-0.5 hover:bg-raised hover:text-ink"
              aria-label="Clear selection"
            >
              <IconClose width={10} height={10} />
            </button>
          </div>
        )}
        <div
          className={cx(
            'flex items-end gap-1.5 rounded-lg border bg-canvas px-2.5 py-2 transition-colors',
            'border-hairline focus-within:border-accent/55',
          )}
        >
          <textarea
            ref={inputRef}
            rows={2}
            value={draft}
            disabled={!ready}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                submit()
              }
            }}
            placeholder={
              !ready
                ? 'Available once the graph is built…'
                : selected
                  ? `Ask about ${selected.name}…`
                  : 'Ask the graph…'
            }
            className="max-h-32 flex-1 resize-none bg-transparent text-xs leading-relaxed text-ink outline-none placeholder:text-faint focus-visible:ring-0 disabled:cursor-not-allowed"
          />
          <Button
            variant="primary"
            size="icon"
            onClick={submit}
            disabled={!draft.trim() || asking || !ready}
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
