/**
 * Pure geometry of the region editor. Boxes are fractions of the image
 * (0–1, the Regional script's X / Y / Width / Height sliders); the mask the
 * script builds clamps a box to the image, so `x + width` may exceed 1 in
 * stored values and is drawn clipped.
 */

export type Box = { x: number; y: number; width: number; height: number }

/** What a drag holds: the whole box, or one edge / corner. */
export type Handle = "move" | "n" | "s" | "e" | "w" | "ne" | "nw" | "se" | "sw"

/** Snap targets per axis (fractions). */
export type Snaps = { x: readonly number[]; y: readonly number[] }

/** Smallest box side a resize leaves. */
export const MIN_SIZE = 0.05

export function clamp(value: number, low: number, high: number): number {
  return Math.min(Math.max(value, low), high)
}

/** `value` on the slider grid `step` (no float noise: 0.1 + 0.2 → 0.3). */
export function quantize(value: number, step: number): number {
  if (!(step > 0)) return value
  return Number((Math.round(value / step) * step).toFixed(6))
}

/** The nearest of `targets` within `threshold` of `value`, else `value`. */
export function snap(value: number, targets: readonly number[], threshold: number): number {
  let best = value
  let distance = threshold
  for (const target of targets) {
    const d = Math.abs(target - value)
    if (d <= distance) {
      best = target
      distance = d
    }
  }
  return best
}

/** Shift that moves the nearer of two edges (`low`, `high`) onto a target, or 0. */
function edgeShift(low: number, high: number, targets: readonly number[], threshold: number): number {
  const toLow = snap(low, targets, threshold) - low
  const toHigh = snap(high, targets, threshold) - high
  if (toLow === 0) return toHigh
  if (toHigh === 0) return toLow
  return Math.abs(toLow) <= Math.abs(toHigh) ? toLow : toHigh
}

export type DragOptions = { step: number; snaps?: Snaps; threshold?: number }

/**
 * `start` after dragging `handle` by `dx`, `dy` (fractions of the image).
 * Moving keeps the size and the box inside the image; resizing moves the
 * held edges, keeping at least MIN_SIZE and the image bounds. Edges snap to
 * `snaps` within `threshold`; results lie on the `step` grid.
 */
export function dragBox(start: Box, handle: Handle, dx: number, dy: number, { step, snaps, threshold = 0 }: DragOptions): Box {
  const targets = snaps ?? { x: [], y: [] }
  if (handle === "move") {
    const width = Math.min(start.width, 1)
    const height = Math.min(start.height, 1)
    let x = clamp(start.x + dx, 0, 1 - width)
    let y = clamp(start.y + dy, 0, 1 - height)
    if (threshold > 0) {
      x = clamp(x + edgeShift(x, x + width, targets.x, threshold), 0, 1 - width)
      y = clamp(y + edgeShift(y, y + height, targets.y, threshold), 0, 1 - height)
    }
    return { x: quantize(x, step), y: quantize(y, step), width: start.width, height: start.height }
  }
  let left = start.x
  let top = start.y
  let right = Math.min(start.x + start.width, 1)
  let bottom = Math.min(start.y + start.height, 1)
  const fit = (value: number, axis: readonly number[]) => (threshold > 0 ? snap(value, axis, threshold) : value)
  if (handle.includes("w")) left = clamp(fit(left + dx, targets.x), 0, right - MIN_SIZE)
  if (handle.includes("e")) right = clamp(fit(right + dx, targets.x), left + MIN_SIZE, 1)
  if (handle.includes("n")) top = clamp(fit(top + dy, targets.y), 0, bottom - MIN_SIZE)
  if (handle.includes("s")) bottom = clamp(fit(bottom + dy, targets.y), top + MIN_SIZE, 1)
  const x = quantize(left, step)
  const y = quantize(top, step)
  return { x, y, width: quantize(right - x, step), height: quantize(bottom - y, step) }
}

export type ArrowKey = "ArrowLeft" | "ArrowRight" | "ArrowUp" | "ArrowDown"

export function isArrowKey(key: string): key is ArrowKey {
  return key === "ArrowLeft" || key === "ArrowRight" || key === "ArrowUp" || key === "ArrowDown"
}

/** `start` after an arrow key: moved by `amount`, or (`resize`) its right / bottom edge moved. */
export function nudgeBox(start: Box, key: ArrowKey, amount: number, resize: boolean, step: number): Box {
  const dx = key === "ArrowLeft" ? -amount : key === "ArrowRight" ? amount : 0
  const dy = key === "ArrowUp" ? -amount : key === "ArrowDown" ? amount : 0
  return dragBox(start, resize ? "se" : "move", dx, dy, { step })
}

/** Snap targets: the image edges and centre, and the edges of `others`. */
export function snapTargets(others: readonly Box[]): Snaps {
  const x = [0, 0.5, 1]
  const y = [0, 0.5, 1]
  for (const box of others) {
    x.push(box.x, Math.min(box.x + box.width, 1))
    y.push(box.y, Math.min(box.y + box.height, 1))
  }
  return { x, y }
}

/** Layouts the toolbar offers: boxes for the first regions (the others are switched off). */
export const LAYOUTS: { label: string; boxes: Box[] }[] = [
  {
    label: "Two columns",
    boxes: [
      { x: 0, y: 0, width: 0.5, height: 1 },
      { x: 0.5, y: 0, width: 0.5, height: 1 },
    ],
  },
  {
    label: "Two rows",
    boxes: [
      { x: 0, y: 0, width: 1, height: 0.5 },
      { x: 0, y: 0.5, width: 1, height: 0.5 },
    ],
  },
  {
    label: "Three columns",
    boxes: [
      { x: 0, y: 0, width: 0.33, height: 1 },
      { x: 0.33, y: 0, width: 0.34, height: 1 },
      { x: 0.67, y: 0, width: 0.33, height: 1 },
    ],
  },
  {
    label: "Three rows",
    boxes: [
      { x: 0, y: 0, width: 1, height: 0.33 },
      { x: 0, y: 0.33, width: 1, height: 0.34 },
      { x: 0, y: 0.67, width: 1, height: 0.33 },
    ],
  },
]
