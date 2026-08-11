import { useEffect, useState } from 'react'
import { Button } from '@/components/common/Button'
import { StatusDot, Tag } from '@/components/common/Bits'
import {
  IconLayers,
  IconMoon,
  IconRepo,
  IconSpark,
  IconSun,
} from '@/components/common/Icons'
import { cx } from '@/lib/format'
import { useAppStore } from '@/store/useAppStore'

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

function ConnectionPill() {
  const connection = useAppStore((s) => s.connection)
  const demo = useAppStore((s) => s.demo)
  const demoPinned = useAppStore((s) => s.demoPinned)
  const setDemoPinned = useAppStore((s) => s.setDemoPinned)
  const bootstrap = useAppStore((s) => s.bootstrap)

  const live = connection === 'online' && !demo

  const label = demo ? 'Sample data' : connection === 'online' ? 'Backend live' : 'Backend offline'
  const tone = demo ? 'accent' : connection === 'online' ? 'signal' : 'danger'

  return (
    <div className="flex items-center gap-1.5">
      <button
        type="button"
        onClick={() => (connection === 'online' ? setDemoPinned(!demoPinned) : bootstrap())}
        title={
          connection === 'online'
            ? demoPinned
              ? 'Switch back to the live backend'
              : 'Pin sample data instead of calling the backend'
            : 'Retry the connection to localhost:8000'
        }
        className={cx(
          'flex h-7 items-center gap-2 rounded-md border border-hairline px-2.5',
          'text-xs text-muted transition-colors hover:border-accent/50 hover:text-ink',
        )}
      >
        <StatusDot tone={tone} pulse={connection === 'checking'} />
        {label}
      </button>
      {live && <Tag tone="signal">:8000</Tag>}
    </div>
  )
}

export function TopBar() {
  const [dark, setDark] = useTheme()
  const inspectorOpen = useAppStore((s) => s.inspectorOpen)
  const toggleInspector = useAppStore((s) => s.toggleInspector)
  const toggleRail = useAppStore((s) => s.toggleRail)
  const view = useAppStore((s) => s.view)
  const setView = useAppStore((s) => s.setView)

  return (
    <header className="z-20 flex h-12 shrink-0 items-center gap-3 border-b border-hairline bg-surface/80 px-3 backdrop-blur">
      {view === 'chat' && (
        <Button size="icon" onClick={toggleRail} title="Toggle repositories" aria-label="Toggle repositories">
          <IconRepo />
        </Button>
      )}

      <div className="flex items-center gap-2.5">
        <div className="grid h-6 w-6 place-items-center rounded bg-accent text-canvas">
          <span className="font-mono text-xs font-bold">R</span>
        </div>
        <div className="leading-none">
          <div className="text-sm font-semibold tracking-tight">Repoint</div>
        </div>
        <span className="hidden text-2xs uppercase tracking-[0.14em] text-faint sm:block">
          Repository Intelligence
        </span>
      </div>

      <div className="ml-4 flex items-center gap-0.5 rounded-md border border-hairline p-0.5">
        {(['chat', 'explorer'] as const).map((id) => (
          <button
            key={id}
            type="button"
            onClick={() => setView(id)}
            className={cx(
              'flex h-6 items-center gap-1.5 rounded px-2.5 text-xs capitalize transition-colors',
              view === id ? 'bg-raised text-ink' : 'text-muted hover:text-ink',
            )}
          >
            {id === 'chat' ? <IconSpark width={12} height={12} /> : <IconLayers width={12} height={12} />}
            {id}
          </button>
        ))}
      </div>

      <div className="ml-auto flex items-center gap-1.5">
        <ConnectionPill />
        <Button
          size="icon"
          onClick={() => setDark(!dark)}
          title={dark ? 'Switch to light' : 'Switch to dark'}
          aria-label="Toggle theme"
        >
          {dark ? <IconSun /> : <IconMoon />}
        </Button>
        {view === 'chat' && (
          <Button
            size="icon"
            onClick={toggleInspector}
            title="Toggle retrieval inspector"
            aria-label="Toggle retrieval inspector"
            className={cx(inspectorOpen && 'bg-raised text-accent')}
          >
            <IconLayers />
          </Button>
        )}
      </div>
    </header>
  )
}
