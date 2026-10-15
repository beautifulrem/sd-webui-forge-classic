/**
 * Geometry of the editor (pure): integer image rectangles and the view
 * transform. Screen coordinates are CSS pixels relative to the editor
 * viewport's top-left corner; `screen = image * scale + offset`.
 */

export type Point = { x: number; y: number }
export type Size = { width: number; height: number }
/** Integer pixel rectangle (`x`, `y` inclusive, `width`/`height` ≥ 0). */
export type Rect = { x: number; y: number; width: number; height: number }
export type View = { scale: number; x: number; y: number }

export const MIN_SCALE = 0.02
export const MAX_SCALE = 32

export const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

export const EMPTY_RECT: Rect = { x: 0, y: 0, width: 0, height: 0 }

export const isEmptyRect = (rect: Rect) => rect.width <= 0 || rect.height <= 0

/** The smallest integer rectangle containing the box `[x0, x1) × [y0, y1)`. */
export function rectFromBounds(x0: number, y0: number, x1: number, y1: number): Rect {
  const left = Math.floor(Math.min(x0, x1))
  const top = Math.floor(Math.min(y0, y1))
  const right = Math.ceil(Math.max(x0, x1))
  const bottom = Math.ceil(Math.max(y0, y1))
  return { x: left, y: top, width: right - left, height: bottom - top }
}

export function unionRect(a: Rect, b: Rect): Rect {
  if (isEmptyRect(a)) return b
  if (isEmptyRect(b)) return a
  return rectFromBounds(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.max(a.x + a.width, b.x + b.width), Math.max(a.y + a.height, b.y + b.height))
}

export function intersectRect(a: Rect, b: Rect): Rect {
  const x0 = Math.max(a.x, b.x)
  const y0 = Math.max(a.y, b.y)
  const x1 = Math.min(a.x + a.width, b.x + b.width)
  const y1 = Math.min(a.y + a.height, b.y + b.height)
  return x1 > x0 && y1 > y0 ? { x: x0, y: y0, width: x1 - x0, height: y1 - y0 } : EMPTY_RECT
}

/** `rect` clipped to an image of `size`. */
export const clipRect = (rect: Rect, size: Size) => intersectRect(rect, { x: 0, y: 0, ...size })

/** Bounding box of a flat `[x0, y0, x1, y1, …]` point list. */
export function polygonBounds(points: ArrayLike<number>): Rect {
  if (points.length < 2) return EMPTY_RECT
  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  for (let i = 0; i + 1 < points.length; i += 2) {
    const x = points[i]!
    const y = points[i + 1]!
    if (x < x0) x0 = x
    if (x > x1) x1 = x
    if (y < y0) y0 = y
    if (y > y1) y1 = y
  }
  return rectFromBounds(x0, y0, x1, y1)
}

export function screenToImage(view: View, point: Point): Point {
  return { x: (point.x - view.x) / view.scale, y: (point.y - view.y) / view.scale }
}

export function imageToScreen(view: View, point: Point): Point {
  return { x: point.x * view.scale + view.x, y: point.y * view.scale + view.y }
}

/** The view that shows the whole image centred in the viewport with `margin` CSS px around it. */
export function fitView(image: Size, viewport: Size, { margin = 16, maxScale = 1 }: { margin?: number; maxScale?: number } = {}): View {
  if (image.width <= 0 || image.height <= 0 || viewport.width <= 0 || viewport.height <= 0) return { scale: 1, x: 0, y: 0 }
  const scale = clamp(Math.min((viewport.width - margin * 2) / image.width, (viewport.height - margin * 2) / image.height, maxScale), MIN_SCALE, MAX_SCALE)
  return centerView(image, viewport, scale)
}

/** The image centred at `scale`. */
export function centerView(image: Size, viewport: Size, scale: number): View {
  return { scale, x: (viewport.width - image.width * scale) / 2, y: (viewport.height - image.height * scale) / 2 }
}

/** Zooms by `factor` keeping the image point under the screen point `at` in place. */
export function zoomAt(view: View, factor: number, at: Point): View {
  const scale = clamp(view.scale * factor, MIN_SCALE, MAX_SCALE)
  const k = scale / view.scale
  return { scale, x: at.x - (at.x - view.x) * k, y: at.y - (at.y - view.y) * k }
}

/**
 * Keeps at least `keep` CSS px of the image inside the viewport, so the
 * image can never be panned out of reach.
 */
export function constrainView(view: View, image: Size, viewport: Size, keep = 48): View {
  const w = image.width * view.scale
  const h = image.height * view.scale
  const minX = Math.min(keep, w) - w
  const minY = Math.min(keep, h) - h
  const maxX = viewport.width - Math.min(keep, w)
  const maxY = viewport.height - Math.min(keep, h)
  return { scale: view.scale, x: clamp(view.x, minX, maxX), y: clamp(view.y, minY, maxY) }
}

/** Wheel delta → zoom factor (trackpads send small deltas, mice ±100 per notch). */
export function wheelZoomFactor(deltaY: number, deltaMode = 0): number {
  const pixels = deltaMode === 1 ? deltaY * 16 : deltaMode === 2 ? deltaY * 400 : deltaY
  return Math.exp(-clamp(pixels, -240, 240) / 300)
}

const distance = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y)
const midpoint = (a: Point, b: Point): Point => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 })

/**
 * Two-finger gesture: `start` is the view when the second finger landed,
 * `from` the two points then and `to` now (screen coordinates). Zooms by the
 * change of the finger distance around the first midpoint and pans with the
 * midpoint.
 */
export function pinchView(start: View, from: readonly [Point, Point], to: readonly [Point, Point]): View {
  const before = distance(from[0], from[1])
  const factor = before > 0 ? distance(to[0], to[1]) / before : 1
  const m0 = midpoint(from[0], from[1])
  const m1 = midpoint(to[0], to[1])
  const zoomed = zoomAt(start, factor, m0)
  return { scale: zoomed.scale, x: zoomed.x + m1.x - m0.x, y: zoomed.y + m1.y - m0.y }
}

/** Zoom steps of the +/− buttons and keys. */
const ZOOM_STEPS = [0.05, 0.1, 0.125, 0.25, 0.333, 0.5, 0.667, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32]

export function nextZoomStep(scale: number, direction: 1 | -1): number {
  if (direction > 0) return ZOOM_STEPS.find((step) => step > scale * 1.001) ?? MAX_SCALE
  return [...ZOOM_STEPS].reverse().find((step) => step < scale * 0.999) ?? MIN_SCALE
}
