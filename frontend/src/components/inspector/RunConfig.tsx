import type { ReactNode } from 'react'
import { Tag } from '@/components/common/Bits'
import { cx } from '@/lib/format'
import { useAppStore } from '@/store/useAppStore'

function Row({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: ReactNode
}) {
  return (
    <div className="px-3 py-3">
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <label className="font-mono text-xs text-ink">{label}</label>
        {hint && <span className="text-2xs text-faint">{hint}</span>}
      </div>
      {children}
    </div>
  )
}

/** Read-only rows for values the backend hardcodes today. */
function Fixed({ value, where }: { value: string; where: string }) {
  return (
    <div className="flex items-center justify-between rounded-md border border-dashed border-hairline px-2.5 py-2">
      <span className="font-mono text-xs text-muted">{value}</span>
      <span className="font-mono text-2xs text-faint">{where}</span>
    </div>
  )
}

export function RunConfig() {
  const settings = useAppStore((s) => s.settings)
  const update = useAppStore((s) => s.updateSettings)

  return (
    <div className="divide-y divide-hairline">
      <Row label="top_k" hint="chunks retrieved per question">
        <div className="flex items-center gap-3">
          <input
            type="range"
            min={1}
            max={20}
            step={1}
            value={settings.topK}
            onChange={(e) => update({ topK: Number(e.target.value) })}
            className="h-1 flex-1 cursor-pointer appearance-none rounded-full bg-hairline accent-accent"
            aria-label="Number of chunks to retrieve"
          />
          <span className="w-7 text-right font-mono text-xs tabular-nums text-accent">
            {settings.topK}
          </span>
        </div>
        <p className="mt-2 text-2xs leading-relaxed text-faint">
          More chunks means broader coverage but a longer prompt — and with fixed-size character
          chunking, more chances to include a fragment cut mid-function.
        </p>
      </Row>

      <Row label="min_relevance" hint="inspector filter only">
        <div className="flex items-center gap-3">
          <input
            type="range"
            min={0}
            max={0.9}
            step={0.05}
            value={settings.minRelevance}
            onChange={(e) => update({ minRelevance: Number(e.target.value) })}
            className="h-1 flex-1 cursor-pointer appearance-none rounded-full bg-hairline accent-accent"
            aria-label="Minimum relevance to display"
          />
          <span className="w-7 text-right font-mono text-xs tabular-nums text-accent">
            {settings.minRelevance.toFixed(2)}
          </span>
        </div>
        <p className="mt-2 text-2xs leading-relaxed text-faint">
          Hides weak chunks from this panel. The model still receives all {settings.topK}.
        </p>
      </Row>

      <Row label="generation model">
        <Fixed value={settings.model} where="llm_service.py" />
      </Row>

      <Row label="embedding model">
        <Fixed value={settings.embedModel} where="embedding_service.py" />
      </Row>

      <Row label="distance space">
        <Fixed value="l2 (chroma default)" where="vector_store.py" />
        <p className="mt-2 text-2xs leading-relaxed text-faint">
          <span className="text-accent">Note:</span> nomic-embed-text is trained for cosine
          similarity. Creating the collection with{' '}
          <code className="font-mono">{'{"hnsw:space": "cosine"}'}</code> would rank these results
          differently.
        </p>
      </Row>

      <div className="px-3 py-3">
        <div className="mb-2 flex flex-wrap gap-1.5">
          <Tag>ollama :11434</Tag>
          <Tag>chroma ./chroma_db</Tag>
          <Tag>collection: repositories</Tag>
        </div>
        <p className={cx('text-2xs leading-relaxed text-faint')}>
          Model and store settings are compiled into the backend. Only{' '}
          <code className="font-mono">top_k</code> travels with the request today.
        </p>
      </div>
    </div>
  )
}
