import { useEffect, useState } from 'react'
import { Button } from '@/components/common/Button'
import { IconWarn } from '@/components/common/Icons'
import { TopBar } from '@/components/layout/TopBar'
import { GraphChat } from '@/components/workspace/GraphChat'
import { GraphPane } from '@/components/workspace/GraphPane'
import { ImportDialog, ImportScreen } from '@/components/workspace/ImportForm'
import { useWorkspaceStore } from '@/store/useWorkspaceStore'

function Offline() {
  const bootstrap = useWorkspaceStore((s) => s.bootstrap)
  return (
    <div className="flex flex-1 items-center justify-center p-8">
      <div className="max-w-sm text-center">
        <IconWarn width={24} height={24} className="mx-auto text-danger" />
        <h2 className="mt-3 text-sm font-semibold text-ink">The backend is not reachable</h2>
        <p className="mt-1.5 text-xs leading-relaxed text-muted">
          Start it with <code className="font-mono">uvicorn app.main:app --port 8000</code> from{' '}
          <code className="font-mono">backend/</code>, then retry.
        </p>
        <Button size="sm" variant="outline" className="mt-4" onClick={() => void bootstrap()}>
          Retry
        </Button>
      </div>
    </div>
  )
}

export default function App() {
  const bootstrap = useWorkspaceStore((s) => s.bootstrap)
  const connection = useWorkspaceStore((s) => s.connection)
  const hasRepositories = useWorkspaceStore((s) => s.repositories.length > 0)
  const [importing, setImporting] = useState(false)

  useEffect(() => {
    void bootstrap()
  }, [bootstrap])

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <TopBar onNewRepository={() => setImporting(true)} />

      {connection === 'offline' ? (
        <Offline />
      ) : connection === 'checking' ? null : !hasRepositories ? (
        <ImportScreen />
      ) : (
        <div className="flex min-h-0 flex-1">
          <main className="min-w-0 flex-1">
            <GraphPane />
          </main>
          <aside className="w-[380px] shrink-0 border-l border-hairline bg-surface/60 xl:w-[420px]">
            <GraphChat />
          </aside>
        </div>
      )}

      {importing && <ImportDialog onClose={() => setImporting(false)} />}
    </div>
  )
}
