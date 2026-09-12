import type { ReactNode } from 'react'
import { cx } from '@/lib/format'

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
