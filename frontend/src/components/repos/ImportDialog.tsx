import { useEffect, useRef, useState } from 'react'
import { Button } from '@/components/common/Button'
import { IconClose } from '@/components/common/Icons'
import { useAppStore } from '@/store/useAppStore'

const SUGGESTIONS = [
  'https://github.com/fastapi/fastapi',
  'https://github.com/encode/httpx',
  'https://github.com/pallets/flask',
]

/** Mirrors the backend's own URL handling, so bad input fails before the POST. */
function validate(raw: string): string | null {
  const value = raw.trim()
  if (!value) return 'Paste a repository URL.'

  let parsed: URL
  try {
    parsed = new URL(value)
  } catch {
    return 'That is not a valid URL.'
  }

  if (parsed.protocol !== 'https:') {
    return 'Only https:// URLs are accepted.'
  }
  if (!/(^|\.)github\.com$/i.test(parsed.hostname)) {
    return 'Only github.com repositories are supported right now.'
  }
  const segments = parsed.pathname.split('/').filter(Boolean)
  if (segments.length < 2) return 'Use the full owner/repo URL.'

  return null
}

export function ImportDialog({ onClose }: { onClose: () => void }) {
  const [url, setUrl] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const importRepository = useAppStore((s) => s.importRepository)

  useEffect(() => {
    inputRef.current?.focus()
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  async function submit() {
    const problem = validate(url)
    if (problem) {
      setError(problem)
      return
    }
    setBusy(true)
    await importRepository(url.trim())
    setBusy(false)
    onClose()
  }

  return (
    <div
      className="fixed inset-0 z-50 grid place-items-center bg-canvas/70 p-4 backdrop-blur-sm"
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Import a repository"
        className="w-full max-w-lg animate-fade-up rounded-xl border border-hairline bg-surface shadow-2xl"
      >
        <div className="flex items-center justify-between border-b border-hairline px-4 py-3">
          <h2 className="text-sm font-semibold">Import a repository</h2>
          <Button size="icon" onClick={onClose} aria-label="Close">
            <IconClose />
          </Button>
        </div>

        <div className="space-y-3 p-4">
          <input
            ref={inputRef}
            value={url}
            onChange={(e) => {
              setUrl(e.target.value)
              if (error) setError(null)
            }}
            onKeyDown={(e) => e.key === 'Enter' && submit()}
            placeholder="https://github.com/owner/repo"
            spellCheck={false}
            className="w-full rounded-lg border border-hairline bg-canvas px-3 py-2.5 font-mono text-sm text-ink placeholder:text-faint focus:border-accent/60"
          />

          {error ? (
            <p className="text-xs text-danger">{error}</p>
          ) : (
            <p className="text-xs text-faint">
              The repository is cloned into <code className="font-mono">backend/repositories/</code>.
              Indexing is a separate step.
            </p>
          )}

          <div className="flex flex-wrap gap-1.5 pt-1">
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
        </div>

        <div className="flex justify-end gap-2 border-t border-hairline px-4 py-3">
          <Button size="sm" onClick={onClose}>
            Cancel
          </Button>
          <Button size="sm" variant="primary" onClick={submit} disabled={busy}>
            {busy ? 'Cloning…' : 'Clone repository'}
          </Button>
        </div>
      </div>
    </div>
  )
}
