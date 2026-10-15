/**
 * `range-slider`: a `[low, high]` pair inside fixed bounds (Gradio's
 * RangeSlider): a two-thumb slider between two number boxes. Generic; used by
 * Sparse Attention's "Timestep Range" and meant for ControlNet / ADetailer.
 *
 * props: { minimum?: number (0), maximum?: number (1), step?: number (0.01) }
 * value: [low, high] with minimum <= low <= high <= maximum. Thumbs snap to
 *        the step grid from `minimum`; typed values are only clamped.
 *
 * Server side: `ui.RangeSlider(name, minimum, maximum, (low, high), step)`
 * (remi/ui/nodes.py), which clamps and orders the pair and keeps integers
 * for integer bounds.
 */
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { NumberField } from "@/components/ui/number-field"
import { Slider } from "@/components/ui/slider"
import { stepDecimals } from "@/ui/values"
import type { WidgetProps } from "@/ui/widgets"
import { registerWidgetCatalogs } from "../_lib/i18n"
import { boundsOf, fromThumbs, normalizeRange, type Range, setEnd } from "./range"

const NS = registerWidgetCatalogs("range-slider", import.meta.glob("./locales/*.json", { eager: true, import: "default" }))

type Props = { minimum?: number; maximum?: number; step?: number }

export default function RangeSlider({ value, onChange, props, disabled, id, label }: WidgetProps<Range | null, Props>) {
  const { t: tw } = useTranslation(NS)
  const bounds = boundsOf(props)
  const range = normalizeRange(value, bounds)
  // While a thumb is dragged the pair stays local; the form gets it once, on commit.
  const [dragged, setDragged] = useState<Range | null>(null)
  const shown = dragged ?? range
  const decimals = stepDecimals(bounds.step)
  const name = label || tw("Range")
  const endLabels = [tw("{{label}}: from", { label: name }), tw("{{label}}: to", { label: name })] as const

  const commit = (next: Range) => {
    if (next[0] !== range[0] || next[1] !== range[1] || !Array.isArray(value)) onChange(next)
  }

  const box = (index: 0 | 1) => (
    <NumberField
      id={index === 0 ? id : undefined}
      aria-label={endLabels[index]}
      size="sm"
      steppers={false}
      className="w-[4.75rem] shrink-0"
      inputClassName="text-right"
      value={shown[index]}
      min={bounds.minimum}
      max={bounds.maximum}
      step={bounds.step}
      format={{ maximumFractionDigits: Math.max(decimals, 0), useGrouping: false }}
      onValueChange={(next) => next !== null && Number.isFinite(next) && commit(setEnd(range, index, next, bounds))}
      disabled={disabled}
    />
  )

  return (
    <div role="group" aria-label={name} className="flex items-center gap-3">
      {box(0)}
      <Slider
        value={shown}
        min={bounds.minimum}
        max={bounds.maximum}
        step={bounds.step}
        minStepsBetweenValues={0}
        getAriaLabel={(index) => endLabels[index === 1 ? 1 : 0]}
        onValueChange={(next) => setDragged(fromThumbs(Array.isArray(next) ? next : [next, range[1]], bounds))}
        onValueCommitted={(next) => {
          setDragged(null)
          commit(fromThumbs(Array.isArray(next) ? next : [next, range[1]], bounds))
        }}
        disabled={disabled}
        className="min-w-0 flex-1 py-1.5"
      />
      {box(1)}
    </div>
  )
}
