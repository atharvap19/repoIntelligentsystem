import { useEffect, useRef, useState } from 'react'
import { Button } from '@/components/common/Button'
import { IconSend } from '@/components/common/Icons'
import { cx } from '@/lib/format'
import { useActiveRepository, useAppStore } from '@/store/useAppStore'

export function Composer() {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const ask = useAppStore((s) => s.ask)
  const asking = useAppStore((s) => s.asking)
  const topK = useAppStore((s) => s.settings.topK)
  const repo = useActiveRepository()

  // Grow with content up to ~7 rows, then scroll.
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 168)}px`
  }, [value])

  // "/" focuses the composer from anywhere, like a code host.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      const typing = target && /^(INPUT|TEXTAREA)$/.test(target.tagName)
      if (e.key === '/' && !typing) {
        e.preventDefault()
        textareaRef.current?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  function submit() {
    const question = value.trim()
    if (!question || asking) return
    setValue('')
    void ask(question)
  }

  const blocked = repo != null && repo.state !== 'ready'

  return (
    <div className="shrink-0 border-t border-hairline bg-surface/70 px-5 py-3 backdrop-blur">
      <div className="mx-auto max-w-3xl">
        {blocked && (
          <p className="mb-2 text-2xs text-faint">
            <span className="font-mono text-accent">{repo?.name}</span> is not indexed — answers will
            come from whatever is already in the vector store.
          </p>
        )}

        <div
          className={cx(
            'flex items-end gap-2 rounded-xl border bg-canvas px-3 py-2 transition-colors',
            'border-hairline focus-within:border-accent/55',
          )}
        >
          <textarea
            ref={textareaRef}
            rows={1}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                submit()
              }
            }}
            placeholder={
              repo ? `Ask about ${repo.name}…` : 'Ask a question about the indexed repository…'
            }
            className="max-h-[168px] flex-1 resize-none bg-transparent py-1.5 text-[0.9375rem] leading-relaxed text-ink outline-none placeholder:text-faint"
          />
          <Button
            variant="primary"
            size="icon"
            className="mb-1 h-8 w-8"
            onClick={submit}
            disabled={!value.trim() || asking}
            title="Send (Enter)"
            aria-label="Send question"
          >
            <IconSend />
          </Button>
        </div>

        <div className="mt-1.5 flex items-center gap-3 px-1 text-2xs text-faint">
          <span>
            <kbd className="font-mono">Enter</kbd> send
          </span>
          <span>
            <kbd className="font-mono">Shift+Enter</kbd> newline
          </span>
          <span>
            <kbd className="font-mono">/</kbd> focus
          </span>
          <span className="ml-auto font-mono">top_k = {topK}</span>
        </div>
      </div>
    </div>
  )
}
