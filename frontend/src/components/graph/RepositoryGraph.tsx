import { useCallback, useEffect, useMemo } from 'react'
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  type Edge,
  type Node,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import { EmptyHint } from '@/components/common/Bits'
import { IconLayers } from '@/components/common/Icons'
import { useGraphStore } from '@/store/useGraphStore'
import type { GraphNode as GraphNodeModel } from '@/lib/graphTypes'
import { nodeTypes, type FlowNodeData } from './GraphNode'

const NODE_WIDTH = 188
const H_GAP = 44
const V_GAP = 132
const FOCUS_Y = 0
const CHILD_Y = 190

/**
 * Lay children out in centred rows beneath the focused node.
 *
 * Deliberately not a force or dagre layout: only one level is on screen at a
 * time, so a deterministic grid is more legible than an organic one and
 * cannot reshuffle when the same view is reloaded.
 */
function layout(focus: GraphNodeModel | null, children: GraphNodeModel[]) {
  const positioned: { model: GraphNodeModel; x: number; y: number }[] = []

  const perRow = Math.max(1, Math.min(6, Math.ceil(Math.sqrt(children.length || 1))))
  const rowWidth = perRow * NODE_WIDTH + (perRow - 1) * H_GAP

  if (focus) {
    positioned.push({ model: focus, x: rowWidth / 2 - NODE_WIDTH / 2, y: FOCUS_Y })
  }

  children.forEach((child, index) => {
    const row = Math.floor(index / perRow)
    const column = index % perRow
    const itemsInRow = Math.min(perRow, children.length - row * perRow)
    const width = itemsInRow * NODE_WIDTH + (itemsInRow - 1) * H_GAP
    const offset = (rowWidth - width) / 2

    positioned.push({
      model: child,
      x: offset + column * (NODE_WIDTH + H_GAP),
      y: CHILD_Y + row * V_GAP,
    })
  })

  return positioned
}

function Canvas() {
  const focus = useGraphStore((s) => s.focus)
  const nodes = useGraphStore((s) => s.nodes)
  const edges = useGraphStore((s) => s.edges)
  const selected = useGraphStore((s) => s.selected)
  const highlight = useGraphStore((s) => s.highlight)
  const drillInto = useGraphStore((s) => s.drillInto)
  const selectNode = useGraphStore((s) => s.selectNode)

  const { fitView } = useReactFlow()

  const flowNodes: Node<FlowNodeData>[] = useMemo(() => {
    const positioned = layout(focus, nodes)
    const highlighting = highlight.origin !== null

    return positioned.map(({ model, x, y }) => {
      const isOrigin = model.id === highlight.origin
      const isDependency = highlight.dependencies.has(model.id)
      const isDependent = highlight.dependents.has(model.id)

      return {
        id: model.id,
        type: 'repoNode',
        position: { x, y },
        draggable: false,
        data: {
          model,
          selected: selected?.id === model.id,
          emphasis: isOrigin
            ? 'origin'
            : isDependency
              ? 'dependency'
              : isDependent
                ? 'dependent'
                : 'none',
          // Dimming is what makes a highlight readable: without it every
          // node still competes for attention.
          dimmed: highlighting && !isOrigin && !isDependency && !isDependent,
        },
      }
    })
  }, [focus, nodes, selected, highlight])

  const flowEdges: Edge[] = useMemo(() => {
    const onScreen = new Set(flowNodes.map((n) => n.id))
    const highlighting = highlight.origin !== null

    // CONTAINS edges from focus to each child, drawn implicitly rather than
    // fetched: the hierarchy is already known from the layout.
    const containment: Edge[] = focus
      ? nodes.map((child) => ({
          id: `contains:${focus.id}->${child.id}`,
          source: focus.id,
          target: child.id,
          type: 'smoothstep',
          animated: false,
          style: {
            stroke: 'rgb(var(--c-hairline))',
            strokeWidth: 1,
            opacity: highlighting ? 0.15 : 0.8,
          },
        }))
      : []

    const relations: Edge[] = edges
      .filter((edge) => onScreen.has(edge.source) && onScreen.has(edge.target))
      .map((edge) => {
        const involved =
          !highlighting ||
          edge.source === highlight.origin ||
          edge.target === highlight.origin
        return {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          type: 'smoothstep',
          animated: involved && highlighting,
          label: edge.kind === 'DEPENDS_ON' ? undefined : undefined,
          style: {
            stroke: involved ? 'rgb(var(--c-accent))' : 'rgb(var(--c-hairline))',
            strokeWidth: involved ? 1.6 : 1,
            opacity: involved ? 0.9 : 0.15,
          },
        }
      })

    return [...containment, ...relations]
  }, [flowNodes, edges, focus, nodes, highlight])

  // Re-frame whenever the level changes, so a drill-down is never off screen.
  useEffect(() => {
    const timer = setTimeout(() => fitView({ padding: 0.18, duration: 320 }), 60)
    return () => clearTimeout(timer)
  }, [focus?.id, nodes.length, fitView])

  const onNodeClick = useCallback(
    (_: unknown, node: Node) => {
      const model = (node.data as FlowNodeData).model
      if (model.id === focus?.id) return
      void selectNode(model)
    },
    [focus?.id, selectNode],
  )

  const onNodeDoubleClick = useCallback(
    (_: unknown, node: Node) => {
      const model = (node.data as FlowNodeData).model
      void drillInto(model)
    },
    [drillInto],
  )

  if (!focus) {
    return (
      <EmptyHint icon={<IconLayers width={22} height={22} />} title="No graph loaded">
        Index a repository to build its structure graph.
      </EmptyHint>
    )
  }

  return (
    <ReactFlow
      nodes={flowNodes}
      edges={flowEdges}
      nodeTypes={nodeTypes}
      onNodeClick={onNodeClick}
      onNodeDoubleClick={onNodeDoubleClick}
      proOptions={{ hideAttribution: true }}
      minZoom={0.15}
      maxZoom={1.8}
      nodesConnectable={false}
      elementsSelectable
      fitView
    >
      <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="rgb(var(--c-hairline))" />
      <Controls showInteractive={false} className="!border-hairline !bg-surface" />
      <MiniMap
        pannable
        zoomable
        className="!border !border-hairline !bg-surface"
        maskColor="rgb(var(--c-canvas) / 0.7)"
        nodeColor={(node) => {
          const kind = (node.data as FlowNodeData).model.kind
          if (kind === 'module' || kind === 'repository') return 'rgb(var(--c-accent))'
          if (kind === 'file') return 'rgb(var(--c-signal))'
          return 'rgb(var(--c-faint))'
        }}
      />
    </ReactFlow>
  )
}

export function RepositoryGraph() {
  return (
    <ReactFlowProvider>
      <Canvas />
    </ReactFlowProvider>
  )
}
