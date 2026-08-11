import { useEffect, useRef } from 'react'
import { Button } from '@/components/common/Button'
import { Tag } from '@/components/common/Bits'
import { IconRefresh, IconSpark } from '@/components/common/Icons'
import { useActiveRepository, useAppStore } from '@/store/useAppStore'
import { Composer } from './Composer'
import { Message } from './Message'

/** Question starters that demonstrate what repo-grounded retrieval is good at. */
const STARTERS = [
  {
    label: 'Entry points',
    question: 'What are the entry points of this codebase and what happens on startup?',
  },
  {
    label: 'Architecture',
    question: 'Describe the overall architecture and how the main modules depend on each other.',
  },
  {
    label: 'Request path',
    question: 'Trace what happens to an incoming request from route to response.',
  },
  {
    label: 'Data models',
    question: 'What are the core data models and where are they defined?',
  },
  {
    label: 'Error handling',
    question: 'How are errors and exceptions handled across the codebase?',
  },
  {
    label: 'Onboarding',
    question: 'If I were joining this project today, which five files should I read first and why?',
  },
]

function Welcome() {
  const ask = useAppStore((s) => s.ask)
  const repo = useActiveRepository()

  return (
    <div className="mx-auto flex max-w-3xl flex-col justify-center px-5 py-12">
      <div className="mb-1.5 flex items-center gap-2 text-accent">
        <IconSpark width={15} height={15} />
        <span className="text-2xs font-semibold uppercase tracking-[0.14em]">
          Grounded in your code
        </span>
      </div>

      <h1 className="text-2xl font-semibold tracking-tight text-ink">
        {repo ? (
          <>
            Ask <span className="font-mono text-accent">{repo.name}</span> anything.
          </>
        ) : (
          'Ask the indexed repository anything.'
        )}
      </h1>

      <p className="mt-2 max-w-[58ch] text-sm leading-relaxed text-muted">
        Every answer is assembled from chunks retrieved out of the vector store — and every one of
        those chunks is shown to you in the inspector, with its file, its rank, and its raw text. No
        hidden context.
      </p>

      <div className="mt-7">
        <div className="mb-2.5 text-2xs uppercase tracking-[0.14em] text-faint">Try one</div>
        <div className="grid gap-2 sm:grid-cols-2">
          {STARTERS.map((starter) => (
            <button
              key={starter.label}
              type="button"
              onClick={() => void ask(starter.question)}
              className="group rounded-lg border border-hairline bg-surface/70 p-3 text-left transition-colors hover:border-accent/45 hover:bg-raised"
            >
              <div className="text-2xs font-semibold uppercase tracking-wider text-accent">
                {starter.label}
              </div>
              <div className="mt-1 text-[0.8125rem] leading-snug text-muted group-hover:text-ink">
                {starter.question}
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

export function ChatPanel() {
  const messages = useAppStore((s) => s.messages)
  const clear = useAppStore((s) => s.clearConversation)
  const repo = useActiveRepository()
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  return (
    <section className="flex min-w-0 flex-1 flex-col">
      <div className="flex h-10 shrink-0 items-center gap-2 border-b border-hairline px-5">
        <span className="text-2xs uppercase tracking-[0.14em] text-faint">asking</span>
        <span className="truncate rounded bg-raised px-1.5 py-0.5 font-mono text-xs text-accent">
          {repo ? repo.name : 'no repository selected'}
        </span>
        {repo?.state === 'ready' && <Tag tone="signal">indexed</Tag>}
        {messages.length > 0 && (
          <Button size="sm" className="ml-auto" onClick={clear} title="Clear conversation">
            <IconRefresh width={13} height={13} />
            New thread
          </Button>
        )}
      </div>

      <div className="scrollbar-thin flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <Welcome />
        ) : (
          <div className="pb-6">
            {messages.map((message) => (
              <Message key={message.id} message={message} />
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      <Composer />
    </section>
  )
}
