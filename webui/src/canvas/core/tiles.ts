/**
 * Tiles of the undo snapshots (pure): an edit keeps the pixels it is about
 * to change, read tile by tile, and only the tiles its dabs touch — a long
 * thin stroke across a large image reads a band, not its whole bounding box.
 */
import { clipRect, intersectRect, isEmptyRect, type Rect, type Size } from "./geometry"

export const TILE = 256

/** The tiles (rects, clipped to the image) that `rect` overlaps. */
export function tilesInRect(rect: Rect, size: Size, tile = TILE): Rect[] {
  const area = clipRect(rect, size)
  if (isEmptyRect(area)) return []
  const out: Rect[] = []
  for (let ty = Math.floor(area.y / tile); ty * tile < area.y + area.height; ty++) {
    for (let tx = Math.floor(area.x / tile); tx * tile < area.x + area.width; tx++) {
      const cell = intersectRect({ x: tx * tile, y: ty * tile, width: tile, height: tile }, area)
      if (!isEmptyRect(cell)) out.push(cell)
    }
  }
  return out
}

/** The tiles touched by dabs `[x, y, radius, …]` (grown by `pad` px), clipped to `bounds`. */
export function tilesForDabs(dabs: ArrayLike<number>, bounds: Rect, pad = 2, tile = TILE): Rect[] {
  const columns = Math.ceil((bounds.x + bounds.width) / tile)
  const keys = new Set<number>()
  for (let i = 0; i + 2 < dabs.length; i += 3) {
    const r = dabs[i + 2]! + pad
    const x0 = Math.max(bounds.x, Math.floor(dabs[i]! - r))
    const x1 = Math.min(bounds.x + bounds.width - 1, Math.ceil(dabs[i]! + r))
    const y0 = Math.max(bounds.y, Math.floor(dabs[i + 1]! - r))
    const y1 = Math.min(bounds.y + bounds.height - 1, Math.ceil(dabs[i + 1]! + r))
    if (x1 < x0 || y1 < y0) continue
    for (let ty = Math.floor(y0 / tile); ty <= Math.floor(y1 / tile); ty++) {
      for (let tx = Math.floor(x0 / tile); tx <= Math.floor(x1 / tile); tx++) keys.add(ty * columns + tx)
    }
  }
  return [...keys]
    .sort((a, b) => a - b)
    .map((key) => intersectRect({ x: (key % columns) * tile, y: Math.floor(key / columns) * tile, width: tile, height: tile }, bounds))
    .filter((cell) => !isEmptyRect(cell))
}
