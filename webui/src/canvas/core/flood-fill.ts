/**
 * Bucket fill region (pure): the 4-connected area around a seed pixel whose
 * colour is within `tolerance` of the seed's (largest channel difference,
 * alpha included, 0–255). Scanline algorithm, O(pixels) with a small stack.
 */
import { EMPTY_RECT, type Rect } from "./geometry"

export type FillRegion = {
  /** 1 for every pixel in the region, row-major `width × height`. */
  mask: Uint8Array
  bounds: Rect
  count: number
}

export function floodFill(rgba: ArrayLike<number>, width: number, height: number, seedX: number, seedY: number, tolerance = 0): FillRegion {
  const mask = new Uint8Array(width * height)
  const sx = Math.floor(seedX)
  const sy = Math.floor(seedY)
  if (sx < 0 || sy < 0 || sx >= width || sy >= height) return { mask, bounds: EMPTY_RECT, count: 0 }

  const s = (sy * width + sx) * 4
  const r0 = rgba[s]!
  const g0 = rgba[s + 1]!
  const b0 = rgba[s + 2]!
  const a0 = rgba[s + 3]!
  const tol = Math.max(0, Math.min(255, tolerance))
  const matches = (index: number) => {
    if (mask[index]) return false
    const j = index * 4
    const a = rgba[j + 3]!
    // Fully transparent pixels match each other whatever their colour bytes.
    if (a0 === 0 && a === 0) return true
    return (
      Math.abs(rgba[j]! - r0) <= tol && Math.abs(rgba[j + 1]! - g0) <= tol && Math.abs(rgba[j + 2]! - b0) <= tol && Math.abs(a - a0) <= tol
    )
  }

  let x0 = sx
  let x1 = sx
  let y0 = sy
  let y1 = sy
  let count = 0
  const stack: number[] = [sx, sy]
  while (stack.length > 0) {
    const y = stack.pop()!
    const x = stack.pop()!
    const row = y * width
    if (!matches(row + x)) continue
    let left = x
    let right = x
    while (left > 0 && matches(row + left - 1)) left--
    while (right < width - 1 && matches(row + right + 1)) right++
    for (let i = left; i <= right; i++) mask[row + i] = 1
    count += right - left + 1
    if (left < x0) x0 = left
    if (right > x1) x1 = right
    if (y < y0) y0 = y
    if (y > y1) y1 = y
    for (const ny of [y - 1, y + 1]) {
      if (ny < 0 || ny >= height) continue
      const nrow = ny * width
      let inSpan = false
      for (let i = left; i <= right; i++) {
        const fits = matches(nrow + i)
        if (fits && !inSpan) {
          stack.push(i, ny)
          inSpan = true
        } else if (!fits) inSpan = false
      }
    }
  }
  return { mask, bounds: { x: x0, y: y0, width: x1 - x0 + 1, height: y1 - y0 + 1 }, count }
}

/** The region's pixels inside `bounds` as opaque RGBA in one colour (transparent elsewhere). */
export function regionPixels(region: FillRegion, imageWidth: number, [r, g, b]: readonly [number, number, number]): Uint8ClampedArray {
  const { bounds, mask } = region
  const out = new Uint8ClampedArray(bounds.width * bounds.height * 4)
  for (let y = 0; y < bounds.height; y++) {
    const src = (bounds.y + y) * imageWidth + bounds.x
    for (let x = 0; x < bounds.width; x++) {
      if (!mask[src + x]) continue
      const j = (y * bounds.width + x) * 4
      out[j] = r
      out[j + 1] = g
      out[j + 2] = b
      out[j + 3] = 255
    }
  }
  return out
}
