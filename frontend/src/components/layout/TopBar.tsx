import { useEffect, useState } from 'react'
import { Button } from '@/components/common/Button'
import { StatusDot } from '@/components/common/Bits'
import { IconMoon, IconPlus, IconSun } from '@/components/common/Icons'
import { cx } from '@/lib/format'
import type { RepositoryEntry } from '@/lib/types'
import { useWorkspaceStore } from '@/store/useWorkspaceStore'

function useTheme() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains('dark'))

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    try {
      localStorage.setItem('repoint.theme', dark ? 'dark' : 'light')
    } catch {
      /* private mode */
    }
  }, [dark])

  return [dark, setDark] as const
}

const STATE_SUFFIX: Record<RepositoryEntry['status']['state'], string> = {
  cloning: ' — cloning…',
  building_graph: ' — building graph…',
  indexing_code: '',
  ready: '',
  error: ' — failed',
}

export function TopBar({ onNewRepository }: { onNewRepository: () => void }) {
  const [dark, setDark] = useTheme()
  const connection = useWorkspaceStore((s) => s.connection)
  const repositories = useWorkspaceStore((s) => s.repositories)
  const active = useWorkspaceStore((s) => s.active)
  const select = useWorkspaceStore((s) => s.select)
  const bootstrap = useWorkspaceStore((s) => s.bootstrap)

  return (
    <header className="z-20 flex h-12 shrink-0 items-center gap-3 border-b border-hairline bg-surface/80 px-3 backdrop-blur">
      <div className="flex items-center gap-2.5">
        <div className="grid h-6 w-6 place-items-center rounded bg-accent text-canvas">
          <span className="font-mono text-xs font-bold">R</span>
        </div>
        <span className="text-sm font-semibold tracking-tight">Repoint</span>
        <span className="hidden text-2xs uppercase tracking-[0.14em] text-faint md:block">
          Knowledge Graph
        </span>
      </div>

      {repositories.length > 0 && (
        <div className="ml-3 flex items-center gap-1.5">
          <select
            value={active ?? ''}
            onChange={(e) => select(e.target.value)}
            aria-label="Repository"
            className="h-7 max-w-[16rem] rounded-md border border-hairline bg-canvas px-2 font-mono text-xs text-ink focus:border-accent/60"
          >
            {repositories.map((entry) => (
              <option key={entry.repository} value={entry.repository}>
                {entry.repository}
                {STATE_SUFFIX[entry.status.state]}
              </option>
            ))}
          </select>
          <Button size="sm" variant="outline" onClick={onNewRepository}>
            <IconPlus width={13} height={13} />
            New
          </Button>
        </div>
      )}

      <div className="ml-auto flex items-center gap-1.5">
        <button
          type="button"
          onClick={() => connection === 'offline' && void bootstrap()}
          title={connection === 'offline' ? 'Retry the connection to localhost:8000' : undefined}
          className={cx(
            'flex h-7 items-center gap-2 rounded-md border border-hairline px-2.5 text-xs text-muted',
            connection === 'offline' && 'hover:border-accent/50 hover:text-ink',
          )}
        >
          <StatusDot
            tone={connection === 'online' ? 'signal' : connection === 'offline' ? 'danger' : 'accent'}
            pulse={connection === 'checking'}
          />
          {connection === 'online' ? 'Backend live' : connection === 'offline' ? 'Backend offline' : 'Connecting'}
        </button>
        <Button
          size="icon"
          onClick={() => setDark(!dark)}
          title={dark ? 'Switch to light' : 'Switch to dark'}
          aria-label="Toggle theme"
        >
          {dark ? <IconSun /> : <IconMoon />}
        </Button>
      </div>
    </header>
  )
}
