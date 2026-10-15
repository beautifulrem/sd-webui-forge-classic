/**
 * Gallery layouts with exact aspect ratios.
 *
 * Justified: rows of images at one height each, filling the width (the last
 * row keeps the target height). Masonry: fixed-width columns, each image as
 * tall as its aspect ratio asks. Both are pure, so the virtualizer only
 * measures rows / tiles that are on screen.
 */

/** Width / height; unknown sizes count as square, extremes are capped so one panorama cannot flatten a row. */
export function aspectOf(width: number | null | undefined, height: number | null | undefined): number {
  if (!width || !height || width <= 0 || height <= 0) return 1
  return Math.min(5, Math.max(0.2, width / height))
}

export type JustifiedRow = {
  /** First item index (inclusive). */
  start: number
  /** Last item index (exclusive). */
  end: number
  height: number
  /** Item widths, in order. */
  widths: number[]
}

/**
 * Greedy justified rows: items are added to a row until, at `targetHeight`,
 * they are at least as wide as the container; the row is then scaled to fit
 * exactly (so full rows are never taller than the target); the last row
 * is left at the target height.
 */
export function justifiedRows(aspects: readonly number[], containerWidth: number, targetHeight: number, gap: number): JustifiedRow[] {
  const rows: JustifiedRow[] = []
  if (containerWidth <= 0 || targetHeight <= 0) return rows
  let start = 0
  let sum = 0
  for (let index = 0; index < aspects.length; index++) {
    sum += aspects[index]!
    const count = index - start + 1
    const available = containerWidth - gap * (count - 1)
    if (sum * targetHeight >= available) {
      const height = available / sum
      rows.push({ start, end: index + 1, height, widths: aspects.slice(start, index + 1).map((aspect) => aspect * height) })
      start = index + 1
      sum = 0
    }
  }
  if (start < aspects.length) {
    const height = targetHeight
    rows.push({ start, end: aspects.length, height, widths: aspects.slice(start).map((aspect) => aspect * height) })
  }
  return rows
}

/** The row holding item `index` (rows are in order). */
export function rowOf(rows: readonly JustifiedRow[], index: number): number {
  let low = 0
  let high = rows.length - 1
  while (low <= high) {
    const middle = (low + high) >> 1
    const row = rows[middle]!
    if (index < row.start) high = middle - 1
    else if (index >= row.end) low = middle + 1
    else return middle
  }
  return -1
}

/**
 * The item above (`-1`) or below (`1`) item `index` in justified rows: the
 * one under the middle of it (the nearest when the row is shorter); null at
 * the first / last row.
 */
export function justifiedNeighbor(rows: readonly JustifiedRow[], index: number, direction: -1 | 1, gap: number): number | null {
  const at = rowOf(rows, index)
  const row = rows[at]
  const next = rows[at + direction]
  if (!row || !next) return null
  let x = 0
  for (let offset = 0; offset < index - row.start; offset++) x += row.widths[offset]! + gap
  const middle = x + row.widths[index - row.start]! / 2
  let left = 0
  for (let offset = 0; offset < next.widths.length; offset++) {
    const right = left + next.widths[offset]!
    if (middle < right + gap / 2) return next.start + offset
    left = right + gap
  }
  return next.end - 1
}

/** How many masonry columns fit (`columnTarget` wide each, at least one). */
export function masonryColumns(containerWidth: number, columnTarget: number, gap: number): { count: number; width: number } {
  const count = Math.max(1, Math.round((containerWidth + gap) / (columnTarget + gap)))
  return { count, width: Math.max(1, (containerWidth - gap * (count - 1)) / count) }
}

/** Thumbnail sizes by the user's tile size choice (CSS pixels). */
export const TILE_SIZES = { s: 140, m: 210, l: 300 } as const
export type TileSize = keyof typeof TILE_SIZES

/** A thumbnail width for a tile of `cssWidth` (device pixels, snapped by the server). */
export function thumbWidth(cssWidth: number, ratio = typeof window === "undefined" ? 1 : window.devicePixelRatio || 1): number {
  return Math.min(1024, Math.ceil(cssWidth * Math.min(ratio, 2)))
}
