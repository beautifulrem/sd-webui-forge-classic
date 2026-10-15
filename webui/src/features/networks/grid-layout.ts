/** Grid / list geometry of the network browser (pure, so it is tested without layout). */
import type { CardSize, ViewMode } from "./store"

export const CARD_MIN_WIDTH: Record<CardSize, number> = { s: 120, m: 164, l: 224 }
export const GAP = 10
export const LIST_ROW = 56
/** At most this many cards are mounted (visible rows plus overscan), so scrolling stays at 60 fps. */
export const MAX_MOUNTED = 60
/** Previews are 3:4 (portrait), like most network previews. */
export const CARD_RATIO = 4 / 3

export type GridLayout = {
  columns: number
  cellWidth: number
  cellHeight: number
  /** Row pitch (cell plus gap). */
  rowHeight: number
  overscan: number
}

export function gridLayout(width: number, height: number, view: ViewMode, size: CardSize): GridLayout {
  const usable = Math.max(0, width)
  if (view === "list") {
    const visible = Math.ceil(Math.max(height, LIST_ROW) / LIST_ROW) + 1
    return { columns: 1, cellWidth: usable, cellHeight: LIST_ROW - 4, rowHeight: LIST_ROW, overscan: clamp(Math.floor((MAX_MOUNTED - visible) / 2), 0, 6) }
  }
  const min = CARD_MIN_WIDTH[size]
  const columns = Math.max(1, Math.floor((usable + GAP) / (min + GAP)))
  const cellWidth = Math.max(48, (usable - GAP * (columns - 1)) / columns)
  const cellHeight = Math.round(cellWidth * CARD_RATIO)
  const rowHeight = cellHeight + GAP
  const visibleRows = Math.ceil(Math.max(height, rowHeight) / rowHeight) + 1
  const overscan = clamp(Math.floor((MAX_MOUNTED / columns - visibleRows) / 2), 0, 2)
  return { columns, cellWidth, cellHeight, rowHeight, overscan }
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

/** The index a key moves the cursor to (`null`: not a navigation key). */
export function moveIndex(key: string, index: number, count: number, columns: number, pageRows: number): number | null {
  if (count === 0) return null
  const at = index < 0 ? 0 : index
  const moves: Record<string, number> = {
    ArrowLeft: at - 1,
    ArrowRight: at + 1,
    ArrowUp: at - columns,
    ArrowDown: at + columns,
    PageUp: at - columns * pageRows,
    PageDown: at + columns * pageRows,
    Home: 0,
    End: count - 1,
  }
  if (!(key in moves)) return null
  if (index < 0) return key === "End" ? count - 1 : 0
  return clamp(moves[key]!, 0, count - 1)
}
