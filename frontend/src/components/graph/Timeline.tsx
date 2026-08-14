import { useEffect, useMemo, useRef, useState } from 'react'
import { Button } from '@/components/common/Button'
import { Tag } from '@/components/common/Bits'
import { IconRefresh } from '@/components/common/Icons'
import { cx } from '@/lib/format'
import type { CommitDot } from '@/lib/graphTypes'
import { useGraphStore } from '@/store/useGraphStore'

const DOT = 9
const MERGE_LANE = 15

function formatDate(seconds: number | null | undefined): string {
  if (!seconds) return '—'
  return new Date(seconds * 1000).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
}

/**
 * Commit history, drawn the way a code host draws it.
 *
 * Phase 3 had a module-activity heatmap with a draggable cursor. It showed
 * *when* modules were busy but never *what happened*, so the answer to "what
 * changed here?" was a date rather than a commit. Part 11 asks for dots on a
 * line: each dot is a real commit, clicking one moves the repository to that
 * state, and hovering one names the author, the message and the file count.
 *
 * Dots are evenly spaced rather than placed by timestamp. Real history is
 * bursty — a week of daily commits then three months of silence — and time
 * spacing collapses the busy weeks into an unclickable smear. Even spacing
 * keeps every commit reachable; the date labels carry the real chronology.
 *
 * Merges sit on a second lane below the trunk, which is the visual convention
 * the brief's sketch shows and reads correctly even without colour.
 */
function CommitTooltip({ commit }: { commit: CommitDot }) {
  return (
    <div className="pointer-events-none absolute bottom-full left-1/2 z-30 mb-2 w-64 -translate-x-1/2 rounded-md border border-hairline bg-surface p-2 shadow-xl">
      <div className="flex items-baseline gap-1.5">
        <span className="font-mono text-2xs text-accent">{commit.short_sha}</span>
        {commit.release && <Tag tone="signal">{commit.release}</Tag>}
        {commit.is_merge && <Tag>merge</Tag>}
      </div>
      <p className="mt-1 line-clamp-3 text-2xs leading-snug text-ink">{commit.summary}</p>
      <div className="mt-1.5 flex items-center gap-2 text-[0.58rem] text-faint">
        <span className="truncate">{commit.author}</span>
        <span>·</span>
        <span className="shrink-0">{formatDate(commit.authored_at)}</span>
        {/* Merge commits legitimately report zero: their file lists are not
            walked, because a merge diff restates the merged branch. */}
        {!commit.is_merge && (
          <>
            <span>·</span>
            <span className="shrink-0">{commit.files_changed} files</span>
          </>
        )}
      </div>
    </div>
  )
}

function ChangedFiles() {
  const commit = useGraphStore((s) => s.commit)
  const revealNode = useGraphStore((s) => s.revealNode)
  const openFile = useGraphStore((s) => s.openFile)

  if (!commit) return null

  return (
    <div className="border-t border-hairline px-3 py-1.5">
      <div className="flex items-baseline gap-2">
        <span className="font-mono text-2xs text-accent">{commit.commit.short_sha}</span>
        <span className="truncate text-2xs text-ink">{commit.commit.summary}</span>
        <span className="ml-auto shrink-0 text-[0.58rem] text-faint">
          {commit.commit.author} · {commit.files_changed} files
        </span>
      </div>
      <div className="mt-1 flex flex-wrap gap-1">
        {commit.files.slice(0, 14).map((file) => (
          <button
            key={file.relative_path}
            type="button"
            disabled={!file.indexed}
            onClick={() => {
              void revealNode(file.node_id)
              void openFile(file.node_id, file.relative_path.split('/').pop(), file.relative_path)
            }}
            title={
              file.indexed
                ? file.relative_path
                : `${file.relative_path} — changed in this commit but not indexed`
            }
            className={cx(
              'rounded border px-1.5 py-0.5 font-mono text-[0.58rem] transition-colors',
              file.indexed
                ? 'border-hairline text-muted hover:border-accent/50 hover:text-accent'
                : 'cursor-default border-hairline/50 text-faint',
            )}
          >
            <span className="mr-1 text-accent">{file.change_type}</span>
            {file.relative_path.split('/').pop()}
          </button>
        ))}
        {commit.files.length > 14 && (
          <span className="self-center text-[0.58rem] text-faint">
            +{commit.files.length - 14} more
          </span>
        )}
      </div>
    </div>
  )
}

export function Timeline() {
  const commits = useGraphStore((s) => s.commits)
  const selected = useGraphStore((s) => s.commit)
  const selectCommit = useGraphStore((s) => s.selectCommit)
  const loading = useGraphStore((s) => s.loading)
  const snapshot = useGraphStore((s) => s.snapshot)
  const [hover, setHover] = useState<string | null>(null)
  const trackRef = useRef<HTMLDivElement>(null)

  const dots = commits?.commits ?? []

  // The newest commit is the interesting end, and it is on the right.
  useEffect(() => {
    const track = trackRef.current
    if (track) track.scrollLeft = track.scrollWidth
  }, [dots.length])

  const trunk = useMemo(() => dots.filter((c) => !c.is_merge), [dots])

  if (!commits || dots.length === 0) {
    return (
      <div className="border-b border-hairline px-3 py-2 text-2xs text-faint">
        No git history was indexed for this repository, so the timeline is unavailable.
      </div>
    )
  }

  const selectedSha = selected?.commit.sha ?? null

  return (
    <div className="shrink-0 border-b border-hairline bg-surface/50">
      <div className="flex items-center gap-2 px-3 pb-1 pt-2">
        <span className="text-2xs uppercase tracking-[0.14em] text-faint">Repository timeline</span>
        <Tag>{commits.range.total} commits</Tag>
        {commits.truncated && <Tag>showing latest {dots.length}</Tag>}
        {selectedSha && (
          <Tag tone="accent">
            viewing {formatDate(selected?.commit.authored_at)}
            {snapshot ? ` · ${snapshot.files_present} files` : ''}
          </Tag>
        )}
        {selectedSha && (
          <Button size="sm" className="ml-auto" onClick={() => void selectCommit(null)}>
            <IconRefresh width={12} height={12} />
            Back to now
          </Button>
        )}
      </div>

      <div
        ref={trackRef}
        className="scrollbar-thin overflow-x-auto px-3 pb-2"
        role="listbox"
        aria-label="Commit history"
      >
        <div
          className="relative"
          style={{ height: 46, minWidth: dots.length * (DOT + 13) + 24 }}
        >
          {/* The trunk line, drawn behind the dots. */}
          <div className="absolute left-0 right-0 top-[13px] h-px bg-hairline" />

          {dots.map((commit, index) => {
            const left = index * (DOT + 13) + 8
            const top = commit.is_merge ? MERGE_LANE + 6 : 9
            const active = commit.sha === selectedSha

            return (
              <div key={commit.sha} className="absolute" style={{ left, top: 0 }}>
                {/* Merge dots hang off the trunk on their own lane. */}
                {commit.is_merge && (
                  <div
                    className="absolute left-1/2 w-px bg-hairline"
                    style={{ top: 13, height: MERGE_LANE - 4 }}
                  />
                )}
                <button
                  type="button"
                  role="option"
                  aria-selected={active}
                  aria-label={`${commit.short_sha} ${commit.summary}`}
                  onClick={() => void selectCommit(active ? null : commit.sha)}
                  onMouseEnter={() => setHover(commit.sha)}
                  onMouseLeave={() => setHover(null)}
                  onFocus={() => setHover(commit.sha)}
                  onBlur={() => setHover(null)}
                  className={cx(
                    'absolute rounded-full border transition-all',
                    active
                      ? 'scale-125 border-accent bg-accent'
                      : commit.release
                        ? 'border-signal bg-signal/70 hover:scale-110'
                        : commit.is_merge
                          ? 'border-hairline bg-surface hover:border-accent/60'
                          : 'border-muted/60 bg-muted/50 hover:scale-110 hover:border-accent',
                    loading && active && 'animate-pulse-dot',
                  )}
                  style={{ width: DOT, height: DOT, top, left: 0 }}
                />
                {hover === commit.sha && <CommitTooltip commit={commit} />}
              </div>
            )
          })}
        </div>

        <div className="flex justify-between font-mono text-[0.58rem] text-faint">
          <span>{formatDate(dots[0]?.authored_at)}</span>
          <span>{trunk.length} commits with file changes</span>
          <span>{formatDate(dots[dots.length - 1]?.authored_at)}</span>
        </div>
      </div>

      <ChangedFiles />
    </div>
  )
}
