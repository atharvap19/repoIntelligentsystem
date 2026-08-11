import { useState } from 'react'
import { Button } from '@/components/common/Button'
import { EmptyHint, StatusDot, Tag } from '@/components/common/Bits'
import { IconBolt, IconClose, IconPlus, IconRepo, IconWarn } from '@/components/common/Icons'
import { cx, formatCount, languageColor, relativeTime } from '@/lib/format'
import { useAppStore } from '@/store/useAppStore'
import type { IndexState, Repository } from '@/lib/types'
import { ImportDialog } from './ImportDialog'

const STATE_META: Record<IndexState, { label: string; tone: 'accent' | 'signal' | 'danger' | 'faint'; busy: boolean }> = {
  unindexed: { label: 'Not indexed', tone: 'faint', busy: false },
  cloning: { label: 'Cloning', tone: 'accent', busy: true },
  indexing: { label: 'Indexing', tone: 'accent', busy: true },
  ready: { label: 'Indexed', tone: 'signal', busy: false },
  error: { label: 'Failed', tone: 'danger', busy: false },
}

/** Stacked bar of the repo's language mix — a glanceable fingerprint. */
function LanguageBar({ languages }: { languages: Record<string, number> }) {
  const entries = Object.entries(languages).sort((a, b) => b[1] - a[1])
  const total = entries.reduce((sum, [, v]) => sum + v, 0) || 1

  return (
    <div className="flex h-1 overflow-hidden rounded-full bg-hairline" aria-hidden="true">
      {entries.map(([lang, value]) => (
        <div
          key={lang}
          title={`${lang} ${Math.round((value / total) * 100)}%`}
          style={{ width: `${(value / total) * 100}%`, background: languageColor(lang) }}
        />
      ))}
    </div>
  )
}

function RepoCard({ repo }: { repo: Repository }) {
  const active = useAppStore((s) => s.activeRepositoryId === repo.id)
  const select = useAppStore((s) => s.selectRepository)
  const index = useAppStore((s) => s.indexRepository)
  const remove = useAppStore((s) => s.removeRepository)

  const meta = STATE_META[repo.state]

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => select(repo.id)}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && select(repo.id)}
      className={cx(
        'group cursor-pointer rounded-lg border p-2.5 transition-colors',
        active
          ? 'border-accent/45 bg-accent/[0.07]'
          : 'border-transparent hover:border-hairline hover:bg-raised',
      )}
    >
      <div className="flex items-center gap-2">
        <StatusDot tone={meta.tone} pulse={meta.busy} />
        <span className="truncate font-mono text-[0.8125rem] font-medium text-ink">{repo.name}</span>
        <Button
          size="icon"
          className="ml-auto opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
          title="Hide from this list (the index is not deleted)"
          aria-label={`Hide ${repo.name}`}
          onClick={(e) => {
            e.stopPropagation()
            remove(repo.id)
          }}
        >
          <IconClose />
        </Button>
      </div>

      <div className="mt-1.5 flex items-center gap-2 pl-3.5 text-2xs text-faint">
        <span>{meta.label}</span>
        <span aria-hidden>·</span>
        <span>{relativeTime(repo.importedAt)}</span>
      </div>

      {repo.languages && (
        <div className="mt-2 pl-3.5">
          <LanguageBar languages={repo.languages} />
        </div>
      )}

      {repo.stats && (
        <div className="mt-2 flex gap-1.5 pl-3.5">
          <Tag>{formatCount(repo.stats.files)} files</Tag>
          <Tag>{formatCount(repo.stats.chunks)} chunks</Tag>
        </div>
      )}

      {repo.state === 'error' && repo.error && (
        <p className="mt-2 flex gap-1.5 pl-3.5 text-2xs leading-relaxed text-danger">
          <IconWarn className="mt-px shrink-0" width={12} height={12} />
          <span className="line-clamp-3">{repo.error}</span>
        </p>
      )}

      {(repo.state === 'unindexed' || repo.state === 'error') && repo.path && (
        <div className="mt-2 pl-3.5">
          <Button
            size="sm"
            variant="outline"
            className="w-full justify-center"
            onClick={(e) => {
              e.stopPropagation()
              index(repo.id)
            }}
          >
            <IconBolt width={13} height={13} />
            {repo.state === 'error' ? 'Retry index' : 'Build index'}
          </Button>
        </div>
      )}
    </div>
  )
}

export function RepoRail() {
  const [importing, setImporting] = useState(false)
  const repositories = useAppStore((s) => s.repositories)

  const indexedCount = repositories.filter((r) => r.state === 'ready').length

  return (
    <>
      <aside className="flex w-[248px] shrink-0 flex-col border-r border-hairline bg-surface/60">
        <div className="flex items-center justify-between px-3 pb-2 pt-3">
          <span className="rail-label px-0">Workspace</span>
          <Button
            size="icon"
            title="Import a repository"
            aria-label="Import a repository"
            onClick={() => setImporting(true)}
          >
            <IconPlus />
          </Button>
        </div>

        <div className="scrollbar-thin flex-1 space-y-0.5 overflow-y-auto px-2 pb-2">
          {repositories.length === 0 ? (
            <EmptyHint icon={<IconRepo width={22} height={22} />} title="No repositories yet">
              Import a GitHub repository to start asking questions about it.
            </EmptyHint>
          ) : (
            repositories.map((repo) => <RepoCard key={repo.id} repo={repo} />)
          )}
        </div>

        <div className="border-t border-hairline px-3 py-2.5 text-2xs text-faint">
          {repositories.length} imported · {indexedCount} indexed
        </div>
      </aside>

      {importing && <ImportDialog onClose={() => setImporting(false)} />}
    </>
  )
}
