import { memo } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { cx, formatCount, languageColor } from '@/lib/format'
import type { GraphNode as GraphNodeModel, NodeKind } from '@/lib/graphTypes'

/** What React Flow carries on each node. */
export interface FlowNodeData extends Record<string, unknown> {
  model: GraphNodeModel
  emphasis: 'none' | 'dependency' | 'dependent' | 'origin'
  dimmed: boolean
  selected: boolean
}

const KIND_LABEL: Record<NodeKind, string> = {
  repository: 'repo',
  module: 'module',
  directory: 'dir',
  file: 'file',
  class: 'class',
  function: 'fn',
  method: 'method',
  external: 'ext',
}

/** Accent per kind, so the hierarchy is readable without reading labels. */
const KIND_ACCENT: Record<NodeKind, string> = {
  repository: 'var(--c-accent)',
  module: 'var(--c-accent)',
  directory: 'rgb(var(--c-muted))',
  file: 'rgb(var(--c-signal))',
  class: 'rgb(var(--c-signal))',
  function: 'rgb(var(--c-muted))',
  method: 'rgb(var(--c-muted))',
  external: 'rgb(var(--c-faint))',
}

const COMPLEXITY_TONE: Record<string, string> = {
  low: 'text-signal',
  medium: 'text-accent',
  high: 'text-danger',
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <span className="flex items-baseline gap-1">
      <span className="font-mono text-[0.62rem] tabular-nums text-ink">{value}</span>
      <span className="text-[0.58rem] uppercase tracking-wide text-faint">{label}</span>
    </span>
  )
}

export const RepoGraphNode = memo(function RepoGraphNode({ data }: NodeProps) {
  const { model, emphasis, dimmed, selected } = data as FlowNodeData
  const kind = model.kind
  const info = model.data
  const isContainer = kind === 'repository' || kind === 'module' || kind === 'directory'

  const accent =
    emphasis === 'dependency'
      ? 'rgb(var(--c-signal))'
      : emphasis === 'dependent'
        ? 'rgb(var(--c-accent))'
        : KIND_ACCENT[kind]

  return (
    <div
      className={cx(
        'w-[188px] rounded-lg border bg-surface px-2.5 py-2 transition-all duration-200',
        'shadow-[0_1px_0_rgb(0_0_0/0.04)]',
        selected ? 'border-accent ring-1 ring-accent/50' : 'border-hairline',
        emphasis === 'origin' && 'border-accent ring-2 ring-accent/60',
        emphasis === 'dependency' && 'border-signal/70',
        emphasis === 'dependent' && 'border-accent/70',
        dimmed && 'opacity-25',
      )}
      style={{ borderLeftColor: accent, borderLeftWidth: 3 }}
    >
      {/* Handles are hidden but required — edges attach to them. */}
      <Handle type="target" position={Position.Top} className="!h-1 !w-1 !border-0 !bg-transparent" />

      <div className="flex items-center gap-1.5">
        <span className="text-[0.58rem] uppercase tracking-[0.12em] text-faint">
          {KIND_LABEL[kind]}
        </span>
        {info.language && (
          <span
            className="h-1.5 w-1.5 shrink-0 rounded-full"
            style={{ background: languageColor(info.language) }}
            title={info.language}
          />
        )}
        {info.parse_failed && (
          <span className="text-[0.58rem] text-danger" title="Static analysis failed">
            !
          </span>
        )}
      </div>

      <div
        className="mt-0.5 truncate font-mono text-[0.8125rem] font-medium text-ink"
        title={model.key}
      >
        {model.name}
      </div>

      <div className="mt-1.5 flex flex-wrap items-baseline gap-x-2.5 gap-y-0.5">
        {isContainer && info.file_count != null && (
          <Metric label="files" value={formatCount(info.file_count)} />
        )}
        {kind === 'file' && info.line_count != null && (
          <Metric label="lines" value={formatCount(info.line_count)} />
        )}
        {kind === 'file' && !!info.function_count && (
          <Metric label="fn" value={info.function_count} />
        )}
        {kind === 'file' && !!info.class_count && (
          <Metric label="cls" value={info.class_count} />
        )}
        {(kind === 'class' || kind === 'function' || kind === 'method') &&
          info.line_count != null && <Metric label="lines" value={info.line_count} />}
        {kind === 'external' && info.uses != null && <Metric label="uses" value={info.uses} />}
      </div>

      {kind === 'file' && info.complexity && (
        <div className="mt-1 text-[0.58rem] uppercase tracking-wide">
          <span className={COMPLEXITY_TONE[info.complexity] ?? 'text-muted'}>
            {info.complexity} complexity
          </span>
        </div>
      )}

      {info.file_count_at != null && info.file_count != null && (
        <div className="mt-1 text-[0.58rem] text-faint">
          {info.file_count_at}/{info.file_count} files at cursor
        </div>
      )}

      <Handle
        type="source"
        position={Position.Bottom}
        className="!h-1 !w-1 !border-0 !bg-transparent"
      />
    </div>
  )
})

export const nodeTypes = { repoNode: RepoGraphNode }
