/**
 * Imperative canvas renderer for the knowledge graph.
 *
 * Canvas rather than SVG: FastAPI's knowledge graph is ~1,500 nodes and
 * ~3,400 edges, and moving that many DOM elements on every simulation tick
 * drops frames. Living outside React also means the draw loop never reads a
 * stale closure — the component pushes state in, the renderer owns the rest.
 */
import {
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
  type Simulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from 'd3-force'
import type { EdgeKind, GraphEdge, GraphNode } from '@/lib/types'

interface SimNode extends SimulationNodeDatum {
  id: string
  model: GraphNode
  radius: number
  degree: number
  community: number
}

interface SimLink extends SimulationLinkDatum<SimNode> {
  source: SimNode
  target: SimNode
  kind: EdgeKind
}

export interface Emphasis {
  ids: Set<string>
  /** Drawn with a heavier ring — the selected node or the answer's subject. */
  focus: string | null
  mode: 'selection' | 'answer'
  /** Label priority, most important first. */
  order: string[]
}

export interface RendererCallbacks {
  onSelect: (nodeId: string | null) => void
  onOpen: (nodeId: string) => void
  onHover: (node: GraphNode | null, x: number, y: number) => void
}

interface View {
  x: number
  y: number
  k: number
}

interface Theme {
  canvas: string
  ink: string
  faint: string
  accent: string
}

const MIN_ZOOM = 0.04
const MAX_ZOOM = 8
const FALLBACK_COLOR = '#8a8a8a'

function cssColor(styles: CSSStyleDeclaration, name: string): string {
  const channels = styles.getPropertyValue(name).trim().split(/\s+/)
  return channels.length === 3 ? `rgb(${channels.join(',')})` : FALLBACK_COLOR
}

function readTheme(): Theme {
  const styles = getComputedStyle(document.documentElement)
  return {
    canvas: cssColor(styles, '--c-canvas'),
    ink: cssColor(styles, '--c-ink'),
    faint: cssColor(styles, '--c-faint'),
    accent: cssColor(styles, '--c-accent'),
  }
}

function radiusFor(kind: GraphNode['kind'], degree: number): number {
  const base = kind === 'file' ? 4.5 : kind === 'class' ? 4 : 2.8
  return Math.min(16, base + Math.sqrt(degree) * 1.25)
}

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2)

export class KnowledgeGraphRenderer {
  private readonly ctx: CanvasRenderingContext2D
  private simulation: Simulation<SimNode, SimLink> | null = null
  private nodes = new Map<string, SimNode>()
  private links: SimLink[] = []
  /** Node ids by degree, for choosing which labels earn space. */
  private byDegree: SimNode[] = []
  private colors: string[] = []

  private emphasis: Emphasis | null = null
  private hidden = new Set<number>()
  private hovered: SimNode | null = null
  private theme: Theme = readTheme()

  private width = 0
  private height = 0
  private view: View = { x: 0, y: 0, k: 1 }
  private animation: { from: View; to: View; start: number } | null = null
  /** Once the user pans, zooms or drags, auto-fit stops fighting them. */
  private userMoved = false
  private dirty = true
  private raf = 0
  private pointer: {
    id: number
    startX: number
    startY: number
    lastX: number
    lastY: number
    node: SimNode | null
    moved: boolean
  } | null = null

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly callbacks: RendererCallbacks,
  ) {
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('Canvas 2D is not available.')
    this.ctx = ctx

    canvas.addEventListener('wheel', this.onWheel, { passive: false })
    canvas.addEventListener('pointerdown', this.onPointerDown)
    canvas.addEventListener('pointermove', this.onPointerMove)
    canvas.addEventListener('pointerup', this.onPointerUp)
    canvas.addEventListener('pointercancel', this.onPointerUp)
    canvas.addEventListener('pointerleave', this.onPointerLeave)
    canvas.addEventListener('dblclick', this.onDoubleClick)

    const loop = (now: number) => {
      this.step(now)
      this.raf = requestAnimationFrame(loop)
    }
    this.raf = requestAnimationFrame(loop)
  }

  destroy() {
    cancelAnimationFrame(this.raf)
    this.simulation?.stop()
    this.canvas.removeEventListener('wheel', this.onWheel)
    this.canvas.removeEventListener('pointerdown', this.onPointerDown)
    this.canvas.removeEventListener('pointermove', this.onPointerMove)
    this.canvas.removeEventListener('pointerup', this.onPointerUp)
    this.canvas.removeEventListener('pointercancel', this.onPointerUp)
    this.canvas.removeEventListener('pointerleave', this.onPointerLeave)
    this.canvas.removeEventListener('dblclick', this.onDoubleClick)
  }

  // -- inputs --------------------------------------------------------------

  /**
   * Replace the graph. With `fresh` false (nodes merged in after an answer),
   * existing nodes keep their positions and new ones start beside their
   * parent, so the layout grows rather than reshuffles.
   */
  setGraph(
    models: GraphNode[],
    edges: GraphEdge[],
    membership: Map<string, number>,
    colors: string[],
    fresh: boolean,
  ) {
    const previous = fresh ? new Map<string, SimNode>() : this.nodes
    this.colors = colors
    if (fresh) {
      this.userMoved = false
      this.animation = null
      this.hovered = null
    }

    const degree = new Map<string, number>()
    for (const edge of edges) {
      if (edge.kind === 'CONTAINS') continue
      degree.set(edge.source, (degree.get(edge.source) ?? 0) + 1)
      degree.set(edge.target, (degree.get(edge.target) ?? 0) + 1)
    }

    const spread = Math.sqrt(models.length || 1) * 14
    const next = new Map<string, SimNode>()
    for (const model of models) {
      const parent = model.parent_id
        ? (next.get(model.parent_id) ?? previous.get(model.parent_id))
        : undefined
      const node: SimNode = previous.get(model.id) ?? {
        id: model.id,
        model,
        radius: 0,
        degree: 0,
        community: -1,
        x: (parent?.x ?? (Math.random() - 0.5) * spread) + (Math.random() - 0.5) * 12,
        y: (parent?.y ?? (Math.random() - 0.5) * spread) + (Math.random() - 0.5) * 12,
      }
      node.model = model
      node.degree = degree.get(model.id) ?? 0
      node.radius = radiusFor(model.kind, node.degree)
      node.community = membership.get(model.id) ?? parent?.community ?? -1
      next.set(model.id, node)
    }

    this.nodes = next
    this.links = edges
      .filter((e) => e.source !== e.target && next.has(e.source) && next.has(e.target))
      .map((e) => ({ source: next.get(e.source)!, target: next.get(e.target)!, kind: e.kind }))
    this.byDegree = [...next.values()].sort((a, b) => b.degree - a.degree)

    this.simulation?.stop()
    const simulation = forceSimulation<SimNode, SimLink>([...next.values()])
      .force(
        'link',
        forceLink<SimNode, SimLink>(this.links)
          .distance((l) => (l.kind === 'CONTAINS' ? 10 + l.target.radius * 2 : 38))
          // Scaled down for hubs, as d3's default does, so a file imported by
          // 200 others is not yanked into a knot.
          .strength((l) =>
            l.kind === 'CONTAINS'
              ? 0.7
              : 0.5 / Math.max(1, Math.min(l.source.degree, l.target.degree)),
          ),
      )
      .force('charge', forceManyBody<SimNode>().strength((n) => -14 - n.radius * 4).distanceMax(420))
      // Weak gravity keeps disconnected pieces from drifting off forever.
      .force('x', forceX<SimNode>(0).strength(0.04))
      .force('y', forceY<SimNode>(0).strength(0.04))
      .force('collide', forceCollide<SimNode>((n) => n.radius + 1.5))
      .alpha(fresh ? 1 : 0.3)
      .alphaDecay(fresh ? 0.022 : 0.035)

    let ticks = 0
    simulation.on('tick', () => {
      ticks++
      if (!this.userMoved && ticks % 3 === 0) this.frame(null, false)
      this.dirty = true
    })
    simulation.on('end', () => {
      if (!this.userMoved) this.frame(null, true)
    })
    this.simulation = simulation
    this.dirty = true
  }

  setEmphasis(emphasis: Emphasis | null) {
    this.emphasis = emphasis
    this.dirty = true
  }

  setHidden(hidden: Set<number>) {
    this.hidden = hidden
    this.dirty = true
  }

  refreshTheme() {
    this.theme = readTheme()
    this.dirty = true
  }

  resize(width: number, height: number) {
    const dpr = window.devicePixelRatio || 1
    this.width = width
    this.height = height
    this.canvas.width = Math.max(1, Math.round(width * dpr))
    this.canvas.height = Math.max(1, Math.round(height * dpr))
    this.canvas.style.width = `${width}px`
    this.canvas.style.height = `${height}px`
    if (!this.userMoved) this.frame(null, false)
    this.dirty = true
  }

  /** Animate onto the emphasised nodes, or the whole graph when none are. */
  frameEmphasis() {
    this.userMoved = true
    this.frame(this.emphasis?.ids ?? null, true)
  }

  fitAll() {
    this.frame(null, true)
  }

  zoomBy(factor: number) {
    this.userMoved = true
    this.zoomAt(this.width / 2, this.height / 2, factor)
  }

  // -- view ----------------------------------------------------------------

  private frame(ids: Set<string> | null, animate: boolean) {
    if (!this.width || !this.height) return
    const members = ids
      ? [...ids].map((id) => this.nodes.get(id)).filter((n): n is SimNode => !!n)
      : [...this.nodes.values()].filter((n) => !this.hidden.has(n.community))
    if (!members.length) return

    let minX = Infinity
    let minY = Infinity
    let maxX = -Infinity
    let maxY = -Infinity
    for (const n of members) {
      const x = n.x ?? 0
      const y = n.y ?? 0
      minX = Math.min(minX, x - n.radius)
      maxX = Math.max(maxX, x + n.radius)
      minY = Math.min(minY, y - n.radius)
      maxY = Math.max(maxY, y + n.radius)
    }

    const padding = ids ? 90 : 40
    const k = Math.max(
      MIN_ZOOM,
      Math.min(
        // A single node is not worth filling the screen with.
        ids ? 2.4 : 1.8,
        (this.width - padding * 2) / Math.max(1, maxX - minX),
        (this.height - padding * 2) / Math.max(1, maxY - minY),
      ),
    )
    const target = {
      k,
      x: this.width / 2 - ((minX + maxX) / 2) * k,
      y: this.height / 2 - ((minY + maxY) / 2) * k,
    }

    if (animate) this.animation = { from: { ...this.view }, to: target, start: performance.now() }
    else this.view = target
    this.dirty = true
  }

  private zoomAt(px: number, py: number, factor: number) {
    const { x, y, k } = this.view
    const next = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, k * factor))
    this.animation = null
    this.view = { k: next, x: px - ((px - x) / k) * next, y: py - ((py - y) / k) * next }
    this.dirty = true
  }

  private step(now: number) {
    if (this.animation) {
      const t = Math.min(1, (now - this.animation.start) / 520)
      const e = ease(t)
      const { from, to } = this.animation
      this.view = {
        k: from.k + (to.k - from.k) * e,
        x: from.x + (to.x - from.x) * e,
        y: from.y + (to.y - from.y) * e,
      }
      if (t >= 1) this.animation = null
      this.dirty = true
    }
    if (!this.dirty) return
    this.dirty = false
    this.draw()
  }

  // -- drawing -------------------------------------------------------------

  private colorOf(node: SimNode) {
    return this.colors[node.community] ?? FALLBACK_COLOR
  }

  private draw() {
    const { ctx, theme, emphasis, hidden } = this
    const dpr = window.devicePixelRatio || 1
    const { x: tx, y: ty, k } = this.view

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, this.width, this.height)
    ctx.save()
    ctx.translate(tx, ty)
    ctx.scale(k, k)

    const visible = (n: SimNode) => !hidden.has(n.community)
    const emphasised = (n: SimNode) => !emphasis || emphasis.ids.has(n.id)

    // Edges, batched into one path per style: thousands of individual
    // strokes per frame is the expensive part of a canvas graph.
    const quiet = new Path2D()
    const loud = new Path2D()
    const byCommunity = new Map<number, Path2D>()
    for (const link of this.links) {
      const { source: s, target: t } = link
      if (!visible(s) || !visible(t)) continue
      let path: Path2D
      if (emphasis) {
        path = emphasis.ids.has(s.id) && emphasis.ids.has(t.id) ? loud : quiet
      } else if (link.kind !== 'CONTAINS' && s.community === t.community && s.community >= 0) {
        path = byCommunity.get(s.community) ?? new Path2D()
        byCommunity.set(s.community, path)
      } else {
        path = quiet
      }
      path.moveTo(s.x ?? 0, s.y ?? 0)
      path.lineTo(t.x ?? 0, t.y ?? 0)
    }

    ctx.lineWidth = 0.8 / k
    ctx.strokeStyle = theme.faint
    ctx.globalAlpha = emphasis ? 0.06 : 0.22
    ctx.stroke(quiet)
    ctx.globalAlpha = 0.42
    for (const [community, path] of byCommunity) {
      ctx.strokeStyle = this.colors[community] ?? FALLBACK_COLOR
      ctx.stroke(path)
    }
    if (emphasis) {
      ctx.globalAlpha = 0.85
      ctx.lineWidth = 1.6 / k
      ctx.strokeStyle = theme.accent
      ctx.stroke(loud)
    }

    // Nodes: dimmed ones first so emphasised ones are never painted over.
    const ordered = emphasis
      ? [...this.nodes.values()].sort((a, b) => Number(emphasised(a)) - Number(emphasised(b)))
      : this.nodes.values()
    for (const node of ordered) {
      if (!visible(node)) continue
      const x = node.x ?? 0
      const y = node.y ?? 0
      const r = node.radius
      const on = emphasised(node)

      ctx.globalAlpha = on ? 1 : 0.13
      ctx.fillStyle = this.colorOf(node)
      ctx.beginPath()
      if (node.model.kind === 'class') {
        // Classes are diamonds, so they read as a different kind of thing
        // than the files and functions around them.
        ctx.moveTo(x, y - r * 1.2)
        ctx.lineTo(x + r * 1.2, y)
        ctx.lineTo(x, y + r * 1.2)
        ctx.lineTo(x - r * 1.2, y)
        ctx.closePath()
      } else {
        ctx.arc(x, y, r, 0, Math.PI * 2)
      }
      ctx.fill()

      if (node.model.kind === 'file') {
        ctx.lineWidth = 1.2 / k
        ctx.strokeStyle = theme.canvas
        ctx.stroke()
      }
      if (emphasis && on) {
        const isFocus = node.id === emphasis.focus
        ctx.lineWidth = (isFocus ? 3 : 1.6) / k
        ctx.strokeStyle = isFocus ? theme.ink : theme.accent
        ctx.beginPath()
        ctx.arc(x, y, r + (isFocus ? 4 : 2.5) / k, 0, Math.PI * 2)
        ctx.stroke()
      }
    }
    if (this.hovered && visible(this.hovered)) {
      ctx.globalAlpha = 1
      ctx.lineWidth = 2 / k
      ctx.strokeStyle = theme.ink
      ctx.beginPath()
      ctx.arc(this.hovered.x ?? 0, this.hovered.y ?? 0, this.hovered.radius + 3 / k, 0, Math.PI * 2)
      ctx.stroke()
    }
    ctx.restore()

    this.drawLabels()
  }

  /** Labels in screen space, so text stays one size at every zoom. */
  private drawLabels() {
    const { ctx, theme, emphasis } = this
    const { x: tx, y: ty, k } = this.view

    const candidates: SimNode[] = []
    const push = (node: SimNode | null | undefined) => {
      if (node && !this.hidden.has(node.community) && !candidates.includes(node)) candidates.push(node)
    }
    push(this.hovered)
    if (emphasis) {
      push(emphasis.focus ? this.nodes.get(emphasis.focus) : null)
      const members = emphasis.order.map((id) => this.nodes.get(id))
      for (const node of members.slice(0, 60)) push(node)
    } else {
      const budget = k < 0.35 ? 10 : k < 0.8 ? 24 : k < 1.6 ? 70 : 160
      for (const node of this.byDegree.slice(0, budget)) push(node)
    }

    ctx.globalAlpha = 1
    ctx.font = '500 11px Inter, ui-sans-serif, system-ui, sans-serif'
    ctx.textBaseline = 'middle'
    ctx.lineJoin = 'round'
    const placed: { x: number; y: number; w: number; h: number }[] = []

    for (const node of candidates) {
      const sx = (node.x ?? 0) * k + tx
      const sy = (node.y ?? 0) * k + ty
      if (sx < -80 || sy < -20 || sx > this.width + 80 || sy > this.height + 20) continue

      const text = node.model.name
      const w = ctx.measureText(text).width
      const box = { x: sx + node.radius * k + 5, y: sy - 8, w: w + 4, h: 16 }
      const important = node === this.hovered || node.id === emphasis?.focus
      if (
        !important &&
        placed.some((p) => box.x < p.x + p.w && box.x + box.w > p.x && box.y < p.y + p.h && box.y + box.h > p.y)
      ) {
        continue
      }
      placed.push(box)

      ctx.lineWidth = 3.5
      ctx.strokeStyle = theme.canvas
      ctx.strokeText(text, box.x, sy)
      ctx.fillStyle = theme.ink
      ctx.fillText(text, box.x, sy)
    }
  }

  // -- interaction ---------------------------------------------------------

  private local(e: MouseEvent) {
    const rect = this.canvas.getBoundingClientRect()
    return { x: e.clientX - rect.left, y: e.clientY - rect.top }
  }

  private nodeAt(sx: number, sy: number): SimNode | null {
    const { x: tx, y: ty, k } = this.view
    const gx = (sx - tx) / k
    const gy = (sy - ty) / k
    let best: SimNode | null = null
    let bestDistance = Infinity
    for (const node of this.nodes.values()) {
      if (this.hidden.has(node.community)) continue
      const dx = (node.x ?? 0) - gx
      const dy = (node.y ?? 0) - gy
      const distance = dx * dx + dy * dy
      const reach = node.radius + 5 / k
      if (distance < reach * reach && distance < bestDistance) {
        best = node
        bestDistance = distance
      }
    }
    return best
  }

  private onWheel = (e: WheelEvent) => {
    e.preventDefault()
    this.userMoved = true
    const { x, y } = this.local(e)
    this.zoomAt(x, y, Math.exp(-e.deltaY * 0.0015))
  }

  private onPointerDown = (e: PointerEvent) => {
    const { x, y } = this.local(e)
    const node = this.nodeAt(x, y)
    this.pointer = { id: e.pointerId, startX: x, startY: y, lastX: x, lastY: y, node, moved: false }
    this.canvas.setPointerCapture(e.pointerId)
  }

  private onPointerMove = (e: PointerEvent) => {
    const { x, y } = this.local(e)
    const pointer = this.pointer

    if (!pointer) {
      const node = this.nodeAt(x, y)
      if (node !== this.hovered) {
        this.hovered = node
        this.dirty = true
      }
      this.canvas.style.cursor = node ? 'pointer' : 'grab'
      this.callbacks.onHover(node?.model ?? null, x, y)
      return
    }

    if (!pointer.moved && Math.hypot(x - pointer.startX, y - pointer.startY) > 3) {
      pointer.moved = true
      this.userMoved = true
      this.animation = null
      this.callbacks.onHover(null, x, y)
      if (pointer.node) this.simulation?.alphaTarget(0.25).restart()
      this.canvas.style.cursor = 'grabbing'
    }
    if (!pointer.moved) return

    const { x: tx, y: ty, k } = this.view
    if (pointer.node) {
      pointer.node.fx = (x - tx) / k
      pointer.node.fy = (y - ty) / k
    } else {
      this.view = { ...this.view, x: tx + x - pointer.lastX, y: ty + y - pointer.lastY }
    }
    pointer.lastX = x
    pointer.lastY = y
    this.dirty = true
  }

  private onPointerUp = (e: PointerEvent) => {
    const pointer = this.pointer
    this.pointer = null
    if (!pointer) return
    if (this.canvas.hasPointerCapture(e.pointerId)) this.canvas.releasePointerCapture(e.pointerId)
    this.canvas.style.cursor = pointer.node ? 'pointer' : 'grab'

    if (pointer.node) {
      pointer.node.fx = null
      pointer.node.fy = null
      if (pointer.moved) this.simulation?.alphaTarget(0)
    }
    if (!pointer.moved) this.callbacks.onSelect(pointer.node?.id ?? null)
  }

  private onPointerLeave = () => {
    if (this.pointer || !this.hovered) return
    this.hovered = null
    this.dirty = true
    this.callbacks.onHover(null, 0, 0)
  }

  private onDoubleClick = (e: MouseEvent) => {
    const { x, y } = this.local(e)
    const node = this.nodeAt(x, y)
    if (node) this.callbacks.onOpen(node.id)
  }
}
