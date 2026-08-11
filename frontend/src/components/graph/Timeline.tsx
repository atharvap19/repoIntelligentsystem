import { useMemo, useState } from 'react'
import { Button } from '@/components/common/Button'
import { Tag } from '@/components/common/Bits'
import { IconRefresh } from '@/components/common/Icons'
import { cx } from '@/lib/format'
import { useGraphStore } from '@/store/useGraphStore'

const LANE_HEIGHT = 22
const MAX_LANES = 7

function formatDate(seconds: number | null | undefined): string {
  if (!seconds) return '—'
  return new Date(seconds * 1000).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
}

/**
 * Module-activity timeline with a draggable cursor.
 *
 * Lanes are modules, not git branches: a clone has one local branch and the
 * branch graph says little about how a codebase grew, whereas module activity
 * says a great deal. Each lane's bars are commits-per-month in that module.
 */
export function Timeline() {
  const timeline = useGraphStore((s) => s.timeline)
  const cursor = useGraphStore((s) => s.cursor)
  const setCursor = useGraphStore((s) => s.setCursor)
  const loading = useGraphStore((s) => s.loading)
  const snapshot = useGraphStore((s) => s.snapshot)
  const [hover, setHover] = useState<number | null>(null)

  const span = timeline?.range
  const start = span?.first_at ?? null
  const end = span?.last_at ?? null

  const lanes = useMemo(() => (timeline?.lanes ?? []).slice(0, MAX_LANES), [timeline])

  const peak = useMemo(() => {
    let max = 1
    for (const lane of lanes) {
      for (const period of lane.periods) max = Math.max(max, period.commits)
    }
    return max
  }, [lanes])

  if (!timeline || !start || !end || end <= start) {
    return (
      <div className="border-b border-hairline px-3 py-2 text-2xs text-faint">
        No git history was indexed for this repository, so the timeline is unavailable.
      </div>
    )
  }

  const total = end - start
  const ratio = (at: number) => Math.max(0, Math.min(1, (at - start) / total))

  function pick(event: React.MouseEvent<HTMLDivElement>) {
    const rect = event.currentTarget.getBoundingClientRect()
    const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width))
    void setCursor(Math.round(start! + fraction * total))
  }

  function track(event: React.MouseEvent<HTMLDivElement>) {
    const rect = event.currentTarget.getBoundingClientRect()
    const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width))
    setHover(Math.round(start! + fraction * total))
  }

  return (
    <div className="shrink-0 border-b border-hairline bg-surface/50">
      <div className="flex items-center gap-2 px-3 pb-1 pt-2">
        <span className="text-2xs uppercase tracking-[0.14em] text-faint">Repository timeline</span>
        <Tag>{timeline.range.total} commits</Tag>
        {timeline.releases.length > 0 && <Tag>{timeline.releases.length} releases</Tag>}
        {cursor !== null && (
          <Tag tone="accent">
            viewing {formatDate(cursor)}
            {snapshot ? ` · ${snapshot.files_present} files` : ''}
          </Tag>
        )}
        {cursor !== null && (
          <Button size="sm" className="ml-auto" onClick={() => void setCursor(null)}>
            <IconRefresh width={12} height={12} />
            Back to now
          </Button>
        )}
      </div>

      <div
        className="relative cursor-crosshair select-none px-3 pb-2"
        onClick={pick}
        onMouseMove={track}
        onMouseLeave={() => setHover(null)}
        role="slider"
        aria-label="Timeline cursor"
        aria-valuemin={start}
        aria-valuemax={end}
        aria-valuenow={cursor ?? end}
        tabIndex={0}
        onKeyDown={(event) => {
          const step = total / 40
          if (event.key === 'ArrowRight') void setCursor(Math.min(end, (cursor ?? end) + step))
          if (event.key === 'ArrowLeft') void setCursor(Math.max(start, (cursor ?? end) - step))
        }}
      >
        <div className="relative" style={{ height: lanes.length * LANE_HEIGHT }}>
          {lanes.map((lane, index) => (
            <div
              key={lane.module}
              className="absolute inset-x-0 flex items-center gap-2"
              style={{ top: index * LANE_HEIGHT, height: LANE_HEIGHT }}
            >
              <span className="w-24 shrink-0 truncate text-right font-mono text-2xs text-muted">
                {lane.module}
              </span>
              <div className="relative h-3.5 flex-1 rounded bg-raised/60">
                {lane.periods.map((period) => {
                  const left = ratio(period.first_at) * 100
                  const width = Math.max(
                    0.7,
                    (ratio(period.last_at) - ratio(period.first_at)) * 100,
                  )
                  const intensity = 0.25 + 0.75 * Math.min(1, period.commits / peak)
                  return (
                    <div
                      key={period.period}
                      className="absolute top-0 h-full rounded-sm bg-accent"
                      style={{ left: `${left}%`, width: `${width}%`, opacity: intensity }}
                      title={`${lane.module} · ${period.period} · ${period.commits} commits`}
                    />
                  )
                })}
              </div>
            </div>
          ))}

          {/* Release markers, drawn across every lane. */}
          {timeline.releases.map((release) => (
            <div
              key={release.name}
              className="pointer-events-none absolute top-0 w-px bg-signal/50"
              style={{ left: `calc(6rem + 0.5rem + ${ratio(release.created_at) * 100}%)`, height: '100%' }}
              title={`${release.name} · ${formatDate(release.created_at)}`}
            />
          ))}

          {hover !== null && (
            <div
              className="pointer-events-none absolute top-0 h-full w-px bg-faint"
              style={{ left: `calc(6rem + 0.5rem + ${ratio(hover) * 100}%)` }}
            />
          )}

          {cursor !== null && (
            <div
              className={cx(
                'pointer-events-none absolute top-0 h-full w-0.5 bg-accent',
                loading && 'animate-pulse-dot',
              )}
              style={{ left: `calc(6rem + 0.5rem + ${ratio(cursor) * 100}%)` }}
            />
          )}
        </div>

        <div className="mt-1 flex justify-between pl-[6.5rem] font-mono text-[0.58rem] text-faint">
          <span>{formatDate(start)}</span>
          <span>{hover !== null ? formatDate(hover) : ''}</span>
          <span>{formatDate(end)}</span>
        </div>
      </div>
    </div>
  )
}
