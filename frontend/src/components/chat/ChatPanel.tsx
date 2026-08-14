import { useEffect, useRef } from 'react'
import { Button } from '@/components/common/Button'
import { Tag } from '@/components/common/Bits'
import { IconRefresh, IconSpark } from '@/components/common/Icons'
import { useActiveRepository, useAppStore } from '@/store/useAppStore'
import { Composer } from './Composer'
import { Message } from './Message'

/** Starters chosen to land on different branches of the router. */
const STARTERS = [
  {
    label: 'Architecture',
    question: 'Describe the overall architecture and how the main modules depend on each other.',
  },
  {
    label: 'Request flow',
    question: 'How does a request flow through this repository?',
  },
  {
    label: 'Hotspots',
    question: 'Which files are most depended on, and which change most often?',
  },
  {
    label: 'Dependencies',
    question: 'What depends on the main entry point?',
  },
  {
    label: 'Layers',
    question: 'Show me the database layer.',
  },
  {
    label: 'Recent history',
    question: 'What changed recently, and in which modules?',
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
        Answers combine three sources — vector search, keyword search, and the repository&rsquo;s
        import graph — so structural questions are answered from static analysis rather than guessed
        from retrieved text. Every chunk that reached the model is shown in the inspector with its
        file, rank and raw text. No hidden context.
      </p>
      <p className="mt-2 max-w-[58ch] text-sm leading-relaxed text-muted">
        Repoint is read-only. It explains repositories; it never writes to them.
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
