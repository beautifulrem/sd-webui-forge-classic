/** Pan / zoom math of the lightbox (pure). Offsets are screen pixels from the viewport centre. */

export type Size = { width: number; height: number }
export type ViewState = { scale: number; x: number; y: number }
export type Point = { x: number; y: number }

export const MIN_SCALE = 0.05
export const MAX_SCALE = 16

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

/** Scale that fits the image inside the viewport, never enlarging it. */
export function fitScale(image: Size, viewport: Size): number {
  if (image.width <= 0 || image.height <= 0 || viewport.width <= 0 || viewport.height <= 0) return 1
  return Math.min(1, viewport.width / image.width, viewport.height / image.height)
}

/** Keeps the image covering the viewport where it is larger, centred where it is smaller. */
export function clampPan(view: ViewState, image: Size, viewport: Size): ViewState {
  const maxX = Math.max(0, (image.width * view.scale - viewport.width) / 2)
  const maxY = Math.max(0, (image.height * view.scale - viewport.height) / 2)
  return { scale: view.scale, x: clamp(view.x, -maxX, maxX), y: clamp(view.y, -maxY, maxY) }
}

/** Zooms by `factor` keeping the image point under `at` (relative to the viewport centre) in place. */
export function zoomAt(view: ViewState, factor: number, at: Point = { x: 0, y: 0 }): ViewState {
  const scale = clamp(view.scale * factor, MIN_SCALE, MAX_SCALE)
  const k = scale / view.scale
  return { scale, x: at.x - (at.x - view.x) * k, y: at.y - (at.y - view.y) * k }
}

/** The initial view: actual size ("zoomed in") or fitted. */
export function initialView(image: Size, viewport: Size, zoomed: boolean): ViewState {
  return { scale: zoomed ? Math.max(1, fitScale(image, viewport)) : fitScale(image, viewport), x: 0, y: 0 }
}

/** Wheel delta → zoom factor (trackpads send small deltas, mice ±100). */
export function wheelFactor(deltaY: number, deltaMode = 0): number {
  const pixels = deltaMode === 1 ? deltaY * 16 : deltaMode === 2 ? deltaY * 400 : deltaY
  return Math.exp(-clamp(pixels, -200, 200) / 300)
}

const distance = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y)
const midpoint = (a: Point, b: Point): Point => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 })

/**
 * Two-finger pinch: `start` is the view when the second finger landed, `from`
 * the two touch points then and `to` now (relative to the viewport centre).
 * Zooms by the change in finger distance around the starting midpoint and
 * pans with the midpoint.
 */
export function pinchView(start: ViewState, from: readonly [Point, Point], to: readonly [Point, Point]): ViewState {
  const before = distance(from[0], from[1])
  const factor = before > 0 ? distance(to[0], to[1]) / before : 1
  const m0 = midpoint(from[0], from[1])
  const m1 = midpoint(to[0], to[1])
  const zoomed = zoomAt(start, factor, m0)
  return { scale: zoomed.scale, x: zoomed.x + m1.x - m0.x, y: zoomed.y + m1.y - m0.y }
}
