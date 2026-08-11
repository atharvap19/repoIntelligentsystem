import { Button } from '@/components/common/Button'
import { EmptyHint } from '@/components/common/Bits'
import { IconBolt, IconFile, IconLayers, IconRepo } from '@/components/common/Icons'
import { formatCount, languageColor, relativeTime } from '@/lib/format'
import type { GraphNode, RelatedNode } from '@/lib/graphTypes'
import { useGraphStore } from '@/store/useGraphStore'

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded border border-hairline px-2 py-1.5">
      <div className="font-mono text-sm tabular-nums text-ink">{value}</div>
      <div className="text-2xs uppercase tracking-wide text-faint">{label}</div>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="border-t border-hairline px-3 py-2.5">
      <div className="mb-1.5 text-2xs uppercase tracking-[0.14em] text-faint">{title}</div>
      {children}
    </div>
  )
}

/** A dependency/dependent row that navigates on click. */
function RelatedRow({ node }: { node: RelatedNode | GraphNode }) {
  const openFile = useGraphStore((s) => s.openFile)
  const focusById = useGraphStore((s) => s.focusById)

  return (
    <button
      type="button"
      onClick={() => {
        void focusById(node.id)
        if (node.kind === 'file') void openFile(node.id, node.name, node.key)
      }}
      className="flex w-full items-center gap-1.5 rounded px-1 py-0.5 text-left transition-colors hover:bg-raised"
      title={node.key}
    >
      {node.data.language && (
        <span
          className="h-1.5 w-1.5 shrink-0 rounded-full"
          style={{ background: languageColor(node.data.language) }}
        />
      )}
      <span className="truncate font-mono text-2xs text-muted">{node.name}</span>
      <span className="ml-auto shrink-0 font-mono text-[0.58rem] text-faint">
        {node.data.line_count ? `${node.data.line_count}L` : ''}
      </span>
    </button>
  )
}

export function FileDetails() {
  const selected = useGraphStore((s) => s.selected)
  const detail = useGraphStore((s) => s.detail)
  const openFile = useGraphStore((s) => s.openFile)
  const showDependencies = useGraphStore((s) => s.showDependencies)
  const clearHighlight = useGraphStore((s) => s.clearHighlight)
  const highlight = useGraphStore((s) => s.highlight)
  const drillInto = useGraphStore((s) => s.drillInto)

  if (!selected) {
    return (
      <EmptyHint icon={<IconLayers width={20} height={20} />} title="Nothing selected">
        Click a node to inspect it. Double-click to expand it.
      </EmptyHint>
    )
  }

  const info = selected.data
  const isFile = selected.kind === 'file'
  const isContainer = ['repository', 'module', 'directory'].includes(selected.kind)
  const highlighting = highlight.origin === selected.id

  return (
    <div className="pb-4">
      <div className="px-3 py-2.5">
        <div className="flex items-center gap-1.5">
          {isFile ? (
            <IconFile width={13} height={13} className="text-faint" />
          ) : (
            <IconRepo width={13} height={13} className="text-faint" />
          )}
          <span className="text-2xs uppercase tracking-[0.14em] text-faint">{selected.kind}</span>
        </div>
        <div className="mt-0.5 break-words font-mono text-sm font-medium text-ink">
          {selected.name}
        </div>
        <div className="mt-0.5 break-all font-mono text-2xs text-faint">{selected.key}</div>

        <div className="mt-2.5 flex flex-wrap gap-1.5">
          {isFile && (
            <Button size="sm" variant="outline" onClick={() => void openFile(selected.id, selected.name, selected.key)}>
              Open code
            </Button>
          )}
          {isContainer && (
            <Button size="sm" variant="outline" onClick={() => void drillInto(selected)}>
              Expand
            </Button>
          )}
          <Button
            size="sm"
            variant={highlighting ? 'primary' : 'outline'}
            onClick={() => (highlighting ? clearHighlight() : void showDependencies(selected.id))}
          >
            <IconBolt width={12} height={12} />
            {highlighting ? 'Clear highlight' : 'Trace dependencies'}
          </Button>
        </div>
      </div>

      <Section title="Metrics">
        <div className="grid grid-cols-2 gap-1.5">
          {info.language && <Stat label="language" value={info.language} />}
          {info.line_count != null && <Stat label="lines" value={formatCount(info.line_count)} />}
          {info.file_count != null && <Stat label="files" value={formatCount(info.file_count)} />}
          {info.class_count != null && <Stat label="classes" value={info.class_count} />}
          {info.function_count != null && <Stat label="functions" value={info.function_count} />}
          {detail?.dependencies && <Stat label="imports" value={detail.dependencies.length} />}
          {detail?.dependents && <Stat label="used by" value={detail.dependents.length} />}
          {info.complexity && <Stat label="complexity" value={info.complexity} />}
          {info.uses != null && <Stat label="uses" value={info.uses} />}
        </div>
        {info.primary_language && (
          <div className="mt-1.5 text-2xs text-faint">
            mostly {info.primary_language}
            {info.languages &&
              Object.keys(info.languages).length > 1 &&
              ` · ${Object.keys(info.languages).length} languages`}
          </div>
        )}
      </Section>

      {detail?.symbols && detail.symbols.length > 0 && (
        <Section title={`Symbols (${detail.symbols.length})`}>
          <div className="max-h-44 space-y-0.5 overflow-y-auto scrollbar-thin">
            {detail.symbols.map((symbol) => (
              <div key={symbol.id} className="flex items-baseline gap-1.5">
                <span className="shrink-0 font-mono text-[0.58rem] uppercase text-faint">
                  {symbol.kind.slice(0, 3)}
                </span>
                <span className="truncate font-mono text-2xs text-muted" title={symbol.name}>
                  {symbol.name}
                </span>
                <span className="ml-auto shrink-0 font-mono text-[0.58rem] text-faint">
                  {symbol.data.start_line}-{symbol.data.end_line}
                </span>
              </div>
            ))}
          </div>
        </Section>
      )}

      {detail?.dependencies && detail.dependencies.length > 0 && (
        <Section title={`Depends on (${detail.dependencies.length})`}>
          <div className="max-h-40 overflow-y-auto scrollbar-thin">
            {detail.dependencies.map((node) => (
              <RelatedRow key={`dep-${node.id}`} node={node} />
            ))}
          </div>
        </Section>
      )}

      {detail?.dependents && detail.dependents.length > 0 && (
        <Section title={`Used by (${detail.dependents.length})`}>
          <div className="max-h-40 overflow-y-auto scrollbar-thin">
            {detail.dependents.map((node) => (
              <RelatedRow key={`dpt-${node.id}`} node={node} />
            ))}
          </div>
        </Section>
      )}

      {detail?.history && detail.history.length > 0 && (
        <Section title="Recent commits">
          <div className="space-y-1.5">
            {detail.history.slice(0, 8).map((commit) => (
              <div key={commit.sha} className="text-2xs">
                <div className="flex items-baseline gap-1.5">
                  <span className="font-mono text-faint">{commit.sha.slice(0, 7)}</span>
                  <span className="text-muted">{commit.author}</span>
                  <span className="ml-auto shrink-0 text-faint">
                    {relativeTime(commit.authored_at * 1000)}
                  </span>
                </div>
                <div className="truncate text-muted" title={commit.summary}>
                  {commit.summary}
                </div>
              </div>
            ))}
          </div>
        </Section>
      )}

      {detail?.history && detail.history.length === 0 && (
        <Section title="Recent commits">
          <p className="text-2xs leading-relaxed text-faint">
            No commits touching this file fall inside the indexed history window.
          </p>
        </Section>
      )}
    </div>
  )
}
