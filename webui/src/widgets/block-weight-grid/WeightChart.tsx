/**
 * The bar chart of the block-weight-grid widget: one bar per block (or
 * layer). Drag across it to paint weights (committed on release), or focus a
 * bar: ↑/↓ adjust by a step (PageUp/PageDown by 0.25), Home/End go to the
 * bounds, ←/→ move to the neighbouring bar.
 */
import { type KeyboardEvent, type PointerEvent, useRef, useState } from "react"
import { cn } from "@/lib/utils"
import { clamp, fmt } from "../_anima/schedule"

const BAND_BAR = ["bg-sky-500/75", "bg-violet-500/75", "bg-amber-500/75", "bg-emerald-500/75"]

type Props = {
  id: string
  weights: readonly number[]
  min: number
  max: number
  step: number
  disabled: boolean
  selected: number
  onSelect: (index: number) => void
  onCommit: (weights: number[]) => void
  /** The depth band of a bar (-1: none). */
  bandOf: (index: number) => number
  /** The block a bar is an inserted copy of, or null. */
  copyOf: (index: number) => number | null
  itemLabel: (index: number) => string
}

const snap = (value: number, step: number) => Number((Math.round(value / step) * step).toFixed(6))

export function WeightChart(props: Props) {
  const { weights, min, max, step, disabled } = props
  const [draft, setDraft] = useState<number[] | null>(null)
  const area = useRef<HTMLDivElement>(null)
  const bars = useRef<(HTMLDivElement | null)[]>([])
  const last = useRef<number | null>(null)
  const shown = draft ?? weights
  const count = weights.length
  const height = (value: number) => `${(clamp((value - min) / (max - min), 0, 1) * 100).toFixed(2)}%`

  const pointAt = (event: PointerEvent): [number, number] | null => {
    const box = area.current?.getBoundingClientRect()
    if (!box || box.width <= 0 || box.height <= 0) return null
    const index = clamp(Math.floor(((event.clientX - box.left) / box.width) * count), 0, count - 1)
    const value = snap(clamp((1 - (event.clientY - box.top) / box.height) * (max - min) + min, min, max), step)
    return [index, value]
  }

  const paint = (event: PointerEvent) => {
    const point = pointAt(event)
    if (!point) return
    const [index, value] = point
    // Fill the bars a fast stroke skipped
    const from = last.current ?? index
    setDraft((current) => {
      const next = [...(current ?? weights)]
      for (let i = Math.min(from, index); i <= Math.max(from, index); i++) next[i] = value
      return next
    })
    last.current = index
    props.onSelect(index)
  }

  const end = () => {
    if (draft) props.onCommit(draft)
    setDraft(null)
    last.current = null
  }

  const onKeyDown = (event: KeyboardEvent, index: number) => {
    const move = (to: number) => {
      const target = clamp(to, 0, count - 1)
      props.onSelect(target)
      bars.current[target]?.focus()
    }
    const set = (value: number) => props.onCommit(weights.map((weight, i) => (i === index ? snap(clamp(value, min, max), step) : weight)))
    const value = weights[index] ?? 1
    const actions: Record<string, () => void> = {
      ArrowUp: () => set(value + step),
      ArrowDown: () => set(value - step),
      PageUp: () => set(value + 0.25),
      PageDown: () => set(value - 0.25),
      Home: () => set(min),
      End: () => set(max),
      ArrowLeft: () => move(index - 1),
      ArrowRight: () => move(index + 1),
    }
    const action = actions[event.key]
    if (!action || disabled) return
    event.preventDefault()
    action()
  }

  const every = count > 40 ? 4 : count > 20 ? 2 : 1

  return (
    <div className="flex flex-col gap-1">
      <div
        ref={area}
        data-testid="weight-chart"
        className={cn("relative flex h-28 touch-none items-end gap-px rounded-lg border bg-muted/30 px-1 pt-1 select-none", disabled ? "opacity-60" : "cursor-crosshair")}
        onPointerDown={(event) => {
          if (disabled || event.button !== 0) return
          event.preventDefault()
          event.currentTarget.setPointerCapture?.(event.pointerId)
          last.current = null
          paint(event)
        }}
        onPointerMove={(event) => event.buttons === 1 && draft && paint(event)}
        onPointerUp={end}
        onPointerCancel={end}
      >
        {min < 1 && max > 1 ? <div aria-hidden className="pointer-events-none absolute inset-x-1 border-t border-dashed border-foreground/30" style={{ bottom: height(1) }} /> : null}
        {shown.map((weight, index) => {
          const band = props.bandOf(index)
          const copy = props.copyOf(index)
          const label = props.itemLabel(index)
          return (
            <div
              key={index}
              ref={(element) => {
                bars.current[index] = element
              }}
              id={index === props.selected ? props.id : undefined}
              role="slider"
              tabIndex={disabled ? -1 : index === props.selected ? 0 : -1}
              aria-label={label}
              aria-valuemin={min}
              aria-valuemax={max}
              aria-valuenow={weight}
              aria-valuetext={fmt(weight, 3)}
              aria-disabled={disabled || undefined}
              title={`${label}: ${fmt(weight, 3)}${copy !== null ? ` (↳ ${copy})` : ""}`}
              className="relative h-full min-w-1 flex-1 rounded-t-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
              onFocus={() => props.onSelect(index)}
              onKeyDown={(event) => onKeyDown(event, index)}
            >
              <div
                className={cn(
                  "absolute inset-x-0 bottom-0 rounded-t-sm transition-[height] duration-75",
                  band >= 0 ? BAND_BAR[band % BAND_BAR.length] : "bg-primary/70",
                  index === props.selected && "ring-2 ring-foreground/60",
                  copy !== null && "opacity-60",
                )}
                style={{ height: height(weight) }}
              />
            </div>
          )
        })}
      </div>
      <div aria-hidden className="flex gap-px px-1 text-[0.6rem] text-muted-foreground tabular-nums">
        {shown.map((_, index) => (
          <span key={index} className="min-w-1 flex-1 text-center">
            {index % every === 0 ? index : ""}
          </span>
        ))}
      </div>
    </div>
  )
}
