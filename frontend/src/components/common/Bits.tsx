import { useEffect, useState, type ReactNode } from 'react'
import { cx } from '@/lib/format'
import { IconCheck, IconCopy } from './Icons'
import { Button } from './Button'

/** Small uppercase metadata pill. */
export function Tag({
  children,
  tone = 'neutral',
  className,
}: {
  children: ReactNode
  tone?: 'neutral' | 'accent' | 'signal' | 'danger'
  className?: string
}) {
  const tones = {
    neutral: 'border-hairline text-muted',
    accent: 'border-accent/40 bg-accent/10 text-accent',
    signal: 'border-signal/40 bg-signal/10 text-signal',
    danger: 'border-danger/40 bg-danger/10 text-danger',
  }
  return (
    <span
      className={cx(
        'inline-flex items-center rounded border px-1.5 py-px font-mono text-2xs',
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  )
}

/** Status dot that pulses only while work is in flight. */
export function StatusDot({
  tone,
  pulse = false,
}: {
  tone: 'accent' | 'signal' | 'danger' | 'faint'
  pulse?: boolean
}) {
  const tones = {
    accent: 'bg-accent',
    signal: 'bg-signal',
    danger: 'bg-danger',
    faint: 'bg-faint',
  }
  return (
    <span
      className={cx('h-1.5 w-1.5 shrink-0 rounded-full', tones[tone], pulse && 'animate-pulse-dot')}
    />
  )
}

/** Three-dot thinking indicator. */
export function Thinking() {
  return (
    <span className="inline-flex items-center gap-1" aria-label="Generating answer">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-1.5 w-1.5 animate-pulse-dot rounded-full bg-accent"
          style={{ animationDelay: `${i * 180}ms` }}
        />
      ))}
    </span>
  )
}

export function CopyButton({ value, label = 'Copy' }: { value: string; label?: string }) {
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (!copied) return
    const t = setTimeout(() => setCopied(false), 1400)
    return () => clearTimeout(t)
  }, [copied])

  return (
    <Button
      size="icon"
      title={copied ? 'Copied' : label}
      aria-label={copied ? 'Copied' : label}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value)
          setCopied(true)
        } catch {
          /* clipboard blocked — nothing useful to say */
        }
      }}
    >
      {copied ? <IconCheck className="text-signal" /> : <IconCopy />}
    </Button>
  )
}

/** Horizontal relevance meter used by every chunk card. */
export function ScoreBar({ value, className }: { value: number; className?: string }) {
  const pct = Math.max(0, Math.min(1, value)) * 100
  return (
    <div className={cx('h-1 overflow-hidden rounded-full bg-hairline', className)}>
      <div
        className="h-full rounded-full bg-gradient-to-r from-signal to-accent transition-[width] duration-500"
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}

export function EmptyHint({
  icon,
  title,
  children,
}: {
  icon: ReactNode
  title: string
  children?: ReactNode
}) {
  return (
    <div className="flex flex-col items-center gap-2 px-6 py-10 text-center">
      <div className="text-faint">{icon}</div>
      <p className="text-sm font-medium text-muted">{title}</p>
      {children && <p className="max-w-[34ch] text-xs leading-relaxed text-faint">{children}</p>}
    </div>
  )
}
