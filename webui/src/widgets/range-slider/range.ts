/** Pure value rules of the `range-slider` widget (mirrors the server's validation). */
import { clamp, snapToStep } from "@/ui/values"

export type Range = [number, number]

export type Bounds = { minimum: number; maximum: number; step: number }

/** Bounds from the node props: `minimum` 0, `maximum` 1, `step` 0.01 unless given (a bad pair falls back too). */
export function boundsOf(props: { minimum?: unknown; maximum?: unknown; step?: unknown }): Bounds {
  const finite = (value: unknown, fallback: number) => (typeof value === "number" && Number.isFinite(value) ? value : fallback)
  let minimum = finite(props.minimum, 0)
  let maximum = finite(props.maximum, 1)
  if (minimum > maximum) [minimum, maximum] = [0, 1]
  const step = finite(props.step, 0.01)
  return { minimum, maximum, step: step > 0 ? step : 0.01 }
}

/** `value` as an ordered pair inside the bounds; anything else is the full range. */
export function normalizeRange(value: unknown, { minimum, maximum }: Bounds): Range {
  if (!Array.isArray(value) || value.length !== 2 || !value.every((v) => typeof v === "number" && Number.isFinite(v))) {
    return [minimum, maximum]
  }
  const [a, b] = (value as number[]).map((v) => clamp(v, minimum, maximum)) as Range
  return a <= b ? [a, b] : [b, a]
}

/** A new low (`index` 0) or high (`index` 1) end: clamped to the bounds and to the other end, so the ends never cross. */
export function setEnd(range: Range, index: 0 | 1, raw: number, { minimum, maximum }: Bounds): Range {
  const next = clamp(raw, minimum, maximum)
  return index === 0 ? [Math.min(next, range[1]), range[1]] : [range[0], Math.max(next, range[0])]
}

/** Slider thumbs → a pair snapped to the step grid from `minimum`. */
export function fromThumbs(thumbs: readonly number[], bounds: Bounds): Range {
  const snapped = thumbs.slice(0, 2).map((v) => clamp(snapToStep(v, bounds.step, bounds.minimum), bounds.minimum, bounds.maximum))
  return normalizeRange(snapped, bounds)
}
