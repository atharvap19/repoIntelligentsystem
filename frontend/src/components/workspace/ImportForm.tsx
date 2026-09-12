import { useEffect, useRef, useState } from 'react'
import { Button } from '@/components/common/Button'
import { IconClose, IconGraph } from '@/components/common/Icons'
import { useWorkspaceStore } from '@/store/useWorkspaceStore'

const SUGGESTIONS = [
  'https://github.com/fastapi/fastapi',
  'https://github.com/encode/httpx',
  'https://github.com/pallets/flask',
]

/** Mirrors the backend's own URL rules, so bad input fails before the POST. */
function validate(raw: string): string | null {
  const value = raw.trim()
  if (!value) return 'Paste a repository URL.'

  let parsed: URL
  try {
    parsed = new URL(value)
  } catch {
    return 'That is not a valid URL.'
  }
  if (parsed.protocol !== 'https:') return 'Only https:// URLs are accepted.'
  if (!/^(www\.)?github\.com$/i.test(parsed.hostname)) {
    return 'Only github.com repositories are supported right now.'
  }
  if (parsed.pathname.split('/').filter(Boolean).length !== 2) {
    return 'Use the repository URL: https://github.com/owner/repo'
  }
  return null
}

function Form({ onDone }: { onDone?: () => void }) {
  const [url, setUrl] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const startImport = useWorkspaceStore((s) => s.startImport)

  useEffect(() => inputRef.current?.focus(), [])

  async function submit() {
    const problem = validate(url)
    if (problem) {
      setError(problem)
      return
    }
    setBusy(true)
    const failure = await startImport(url.trim())
    setBusy(false)
    if (failure) setError(failure)
    else onDone?.()
  }

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        <input
          ref={inputRef}
          value={url}
          onChange={(e) => {
            setUrl(e.target.value)
            if (error) setError(null)
          }}
          onKeyDown={(e) => e.key === 'Enter' && void submit()}
          placeholder="https://github.com/owner/repo"
          spellCheck={false}
          className="min-w-0 flex-1 rounded-lg border border-hairline bg-canvas px-3 py-2.5 font-mono text-sm text-ink placeholder:text-faint focus:border-accent/60"
        />
        <Button variant="primary" onClick={() => void submit()} disabled={busy}>
          {busy ? 'Starting…' : 'Build graph'}
        </Button>
      </div>

      {error ? (
        <p className="text-xs text-danger">{error}</p>
      ) : (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-2xs text-faint">Try</span>
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => {
                setUrl(s)
                setError(null)
              }}
              className="rounded border border-hairline px-2 py-1 font-mono text-2xs text-muted transition-colors hover:border-accent/50 hover:text-ink"
            >
              {s.replace('https://github.com/', '')}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

/** First run: nothing imported yet, so the URL box is the whole screen. */
export function ImportScreen() {
  return (
    <div className="flex flex-1 items-center justify-center p-6">
      <div className="w-full max-w-xl">
        <IconGraph width={30} height={30} className="text-accent" />
        <h1 className="mt-4 text-xl font-semibold tracking-tight text-ink">
          Turn a repository into a knowledge graph
        </h1>
        <p className="mb-6 mt-2 text-sm leading-relaxed text-muted">
          Paste a GitHub URL. Repoint clones it, maps every file, class and function and how they
          import, call and inherit from each other — then you can talk to the graph, and it lights
          up whatever each answer is about.
        </p>
        <Form />
      </div>
    </div>
  )
}

export function ImportDialog({ onClose }: { onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div
      className="fixed inset-0 z-50 grid place-items-center bg-canvas/70 p-4 backdrop-blur-sm"
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Import a repository"
        className="w-full max-w-xl animate-fade-up rounded-xl border border-hairline bg-surface shadow-2xl"
      >
        <div className="flex items-center justify-between border-b border-hairline px-4 py-3">
          <h2 className="text-sm font-semibold">New knowledge graph</h2>
          <Button size="icon" onClick={onClose} aria-label="Close">
            <IconClose />
          </Button>
        </div>
        <div className="p-4">
          <Form onDone={onClose} />
        </div>
      </div>
    </div>
  )
}
