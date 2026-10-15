/**
 * The sampling timeline of the stage-timeline widget: the three stage
 * windows at the current Shift, the strength the run applies (curve area),
 * blocked windows hatched, and draggable start / peak / end handles
 * (←/→ move by 0.01, Shift+←/→ by 0.05, Home/End to the bounds).
 */
import { type KeyboardEvent, type PointerEvent, useId, useRef } from "react"
import { cn } from "@/lib/utils"
import { clamp, curvePath, fmt, type StageKind, stageWindow, type Window } from "../_anima/schedule"

const KIND_FILL: Record<StageKind, string> = {
  composition: "fill-sky-500/15",
  character: "fill-violet-500/15",
  style: "fill-amber-500/15",
}
const KIND_TEXT: Record<StageKind, string> = {
  composition: "text-sky-600 dark:text-sky-400",
  character: "text-violet-600 dark:text-violet-400",
  style: "text-amber-600 dark:text-amber-400",
}

export type Handle = "start" | "peak" | "end"

type Props = {
  shift: number
  /** Stage labels by kind (translated). */
  stageLabels: Record<StageKind, string>
  /** The strength multiplier at a progress (0..1). */
  factor: (progress: number) => number
  /** Blocked windows (drawn hatched). */
  bans: readonly Window[]
  window: Window
  /** Handles are shown and draggable (else the window is derived). */
  editable: boolean
  disabled?: boolean
  handleLabels?: Record<Handle, string>
  onWindow?: (window: Window) => void
  /** Axis tick text of a progress. */
  tick: (progress: number) => string
  /** Summary for screen readers. */
  summary: string
  className?: string
}

const W = 1000
const H = 100
const TICKS = [0, 0.25, 0.5, 0.75, 1]
const KINDS: StageKind[] = ["composition", "character", "style"]

export function Timeline(props: Props) {
  const area = useRef<HTMLDivElement>(null)
  const hatch = `${useId()}-hatch`
  const dragging = useRef<Handle | null>(null)
  const { window: win } = props
  const top = Math.max(2, ...Array.from({ length: 41 }, (_, i) => props.factor(i / 40)))
  const area01 = (x: number) => clamp(props.factor(x) / top, 0, 1)
  const path = curvePath(area01, W, H)
  const one = H - (1 / top) * H

  const move = (handle: Handle, to: number) => {
    const p = Number(clamp(to, 0, 1).toFixed(4))
    if (handle === "start") props.onWindow?.({ start: Math.min(p, win.end), end: win.end, peak: clamp(win.peak, Math.min(p, win.end), win.end) })
    else if (handle === "end") props.onWindow?.({ start: win.start, end: Math.max(p, win.start), peak: clamp(win.peak, win.start, Math.max(p, win.start)) })
    else props.onWindow?.({ ...win, peak: clamp(p, win.start, win.end) })
  }

  const progressAt = (event: PointerEvent) => {
    const box = area.current?.getBoundingClientRect()
    return box && box.width > 0 ? (event.clientX - box.left) / box.width : null
  }

  const onKeyDown = (event: KeyboardEvent, handle: Handle) => {
    const by = event.shiftKey ? 0.05 : 0.01
    const keys: Record<string, number> = { ArrowLeft: win[handle] - by, ArrowDown: win[handle] - by, ArrowRight: win[handle] + by, ArrowUp: win[handle] + by, Home: 0, End: 1 }
    if (!(event.key in keys) || props.disabled) return
    event.preventDefault()
    move(handle, keys[event.key]!)
  }

  const handle = (name: Handle) => {
    const at = win[name]
    const label = props.handleLabels?.[name] ?? name
    return (
      <div
        key={name}
        role="slider"
        tabIndex={props.disabled ? -1 : 0}
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={Number(at.toFixed(4))}
        aria-valuetext={`${fmt(at, 3)} · ${props.tick(at)}`}
        aria-disabled={props.disabled || undefined}
        data-handle={name}
        title={`${label}: ${fmt(at, 3)}`}
        className={cn(
          "group/handle absolute top-0 bottom-0 z-10 -ml-2 flex w-4 justify-center outline-none",
          props.disabled ? "pointer-events-none" : "cursor-ew-resize",
        )}
        style={{ left: `${at * 100}%` }}
        onPointerDown={(event) => {
          if (props.disabled || event.button !== 0) return
          event.preventDefault()
          event.stopPropagation()
          event.currentTarget.setPointerCapture?.(event.pointerId)
          dragging.current = name
        }}
        onPointerMove={(event) => {
          if (dragging.current !== name) return
          const p = progressAt(event)
          if (p !== null) move(name, p)
        }}
        onPointerUp={() => (dragging.current = null)}
        onPointerCancel={() => (dragging.current = null)}
        onKeyDown={(event) => onKeyDown(event, name)}
      >
        {name === "peak" ? (
          <span className="mt-0.5 size-3 rotate-45 rounded-[2px] border-2 border-background bg-primary shadow group-focus-visible/handle:ring-2 group-focus-visible/handle:ring-ring" />
        ) : (
          <span className="h-full w-0.5 rounded bg-foreground/70 group-hover/handle:bg-foreground group-focus-visible/handle:w-1 group-focus-visible/handle:bg-ring" />
        )}
      </div>
    )
  }

  return (
    <div className={cn("flex flex-col gap-1", props.className)}>
      <div ref={area} className="relative h-28 rounded-lg border bg-muted/20 select-none" data-testid="stage-timeline-plot">
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label={props.summary} className="absolute inset-0 h-full w-full overflow-visible">
          <defs>
            <pattern id={hatch} width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1="0" y1="0" x2="0" y2="8" className="stroke-destructive/50" strokeWidth="3" />
            </pattern>
          </defs>
          {KINDS.map((kind) => {
            const band = stageWindow(kind, props.shift)
            return <rect key={kind} x={band.start * W} y={0} width={(band.end - band.start) * W} height={H} className={KIND_FILL[kind]} />
          })}
          {props.bans.map((ban, i) => (
            <rect key={i} x={ban.start * W} y={0} width={Math.max(ban.end - ban.start, 0.002) * W} height={H} fill={`url(#${hatch})`} data-testid="stage-ban" />
          ))}
          <line x1={0} x2={W} y1={one} y2={one} strokeDasharray="6 6" className="stroke-foreground/25" vectorEffect="non-scaling-stroke" />
          <path d={`${path}L${W},${H}L0,${H}Z`} className="fill-primary/20" />
          <path d={path} fill="none" strokeWidth={2} className="stroke-primary" vectorEffect="non-scaling-stroke" data-testid="stage-strength" />
        </svg>
        {KINDS.map((kind) => {
          const band = stageWindow(kind, props.shift)
          return (
            <span
              key={kind}
              aria-hidden
              className={cn("pointer-events-none absolute top-0.5 -translate-x-1/2 truncate text-[0.625rem] font-medium", KIND_TEXT[kind])}
              style={{ left: `${((band.start + band.end) / 2) * 100}%`, maxWidth: `${(band.end - band.start) * 100}%` }}
            >
              {props.stageLabels[kind]}
            </span>
          )
        })}
        {props.editable ? (["start", "end", "peak"] as const).map(handle) : null}
      </div>
      <div aria-hidden className="relative h-4 text-[0.625rem] text-muted-foreground tabular-nums">
        {TICKS.map((at) => (
          <span key={at} className={cn("absolute", at === 0 ? "left-0" : at === 1 ? "right-0" : "-translate-x-1/2")} style={at > 0 && at < 1 ? { left: `${at * 100}%` } : undefined}>
            {props.tick(at)}
          </span>
        ))}
      </div>
    </div>
  )
}
