import { useEffect } from 'react'
import { TopBar } from '@/components/layout/TopBar'
import { RepoRail } from '@/components/repos/RepoRail'
import { ChatPanel } from '@/components/chat/ChatPanel'
import { RetrievalInspector } from '@/components/inspector/RetrievalInspector'
import { ExplorerView } from '@/components/graph/ExplorerView'
import { useAppStore } from '@/store/useAppStore'

export default function App() {
  const bootstrap = useAppStore((s) => s.bootstrap)
  const railOpen = useAppStore((s) => s.railOpen)
  const inspectorOpen = useAppStore((s) => s.inspectorOpen)
  const view = useAppStore((s) => s.view)

  useEffect(() => {
    void bootstrap()
  }, [bootstrap])

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <TopBar />
      {/* Both views stay mounted-by-branch rather than routed: the Phase 2
          chat is untouched, and Explorer is additive. */}
      {view === 'explorer' ? (
        <ExplorerView />
      ) : (
        <div className="flex min-h-0 flex-1">
          {railOpen && <RepoRail />}
          <ChatPanel />
          {inspectorOpen && <RetrievalInspector />}
        </div>
      )}
    </div>
  )
}
