/** Presentation helpers shared across panels. */

/**
 * ChromaDB returns a raw distance, and the collection currently uses the
 * default L2 space — so the number is unbounded and means nothing to a reader.
 * Map it onto a 0–1 "relevance" purely for display; it is a monotonic
 * rescaling, never a probability.
 */
export function toRelevance(distance: number): number {
  if (!Number.isFinite(distance) || distance < 0) return 0
  return 1 / (1 + distance)
}

export function relevanceLabel(distance: number): string {
  const r = toRelevance(distance)
  if (r >= 0.72) return 'strong'
  if (r >= 0.55) return 'likely'
  if (r >= 0.42) return 'weak'
  return 'tenuous'
}

export function formatMs(ms?: number): string {
  if (ms == null || !Number.isFinite(ms)) return '—'
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)}s`
}

export function formatCount(n?: number): string {
  if (n == null) return '—'
  if (n < 1000) return String(n)
  if (n < 1_000_000) return `${(n / 1000).toFixed(n < 10_000 ? 1 : 0)}k`
  return `${(n / 1_000_000).toFixed(1)}M`
}

export function fileName(path: string): string {
  const parts = path.split(/[\\/]/)
  return parts[parts.length - 1] || path
}

export function dirName(path: string): string {
  const parts = path.split(/[\\/]/)
  parts.pop()
  return parts.join('/')
}

/** Collapse a long path to fit a narrow rail: `app/…/rag/parser.py`. */
export function truncatePath(path: string, maxSegments = 3): string {
  const parts = path.split(/[\\/]/).filter(Boolean)
  if (parts.length <= maxSegments) return parts.join('/')
  return [parts[0], '…', ...parts.slice(-(maxSegments - 1))].join('/')
}

const EXTENSION_LANGUAGE: Record<string, string> = {
  py: 'python',
  js: 'javascript',
  jsx: 'javascript',
  ts: 'typescript',
  tsx: 'typescript',
  java: 'java',
  c: 'c',
  cpp: 'cpp',
  cs: 'csharp',
  go: 'go',
  rs: 'rust',
  kt: 'kotlin',
  sql: 'sql',
  html: 'xml',
  css: 'css',
  md: 'markdown',
  json: 'json',
  yml: 'yaml',
  yaml: 'yaml',
}

export function languageOf(path: string, declared?: string): string {
  if (declared && declared !== 'unknown') {
    return declared === 'html' ? 'xml' : declared
  }
  const ext = path.split('.').pop()?.toLowerCase() ?? ''
  return EXTENSION_LANGUAGE[ext] ?? 'plaintext'
}

/** Stable amber→teal hues so a language keeps its colour across every panel. */
const LANGUAGE_HUES: Record<string, string> = {
  python: '38 92% 58%',
  javascript: '48 92% 60%',
  typescript: '206 78% 58%',
  java: '18 82% 58%',
  go: '188 72% 52%',
  rust: '12 74% 56%',
  c: '224 44% 60%',
  cpp: '232 52% 62%',
  csharp: '270 56% 64%',
  kotlin: '284 62% 62%',
  sql: '160 54% 48%',
  html: '14 84% 60%',
  xml: '14 84% 60%',
  css: '250 68% 66%',
  markdown: '0 0% 60%',
  plaintext: '0 0% 48%',
}

export function languageColor(language: string): string {
  return `hsl(${LANGUAGE_HUES[language] ?? LANGUAGE_HUES.plaintext})`
}

export function relativeTime(ts: number, now = Date.now()): string {
  const s = Math.max(0, Math.round((now - ts) / 1000))
  if (s < 45) return 'just now'
  if (s < 3600) return `${Math.round(s / 60)}m ago`
  if (s < 86400) return `${Math.round(s / 3600)}h ago`
  return `${Math.round(s / 86400)}d ago`
}

export function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(' ')
}
