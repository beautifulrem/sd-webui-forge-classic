/**
 * Multi-selection over an ordered list of ids: click toggles (or replaces,
 * without a modifier outside selection mode), Shift extends from the anchor,
 * Ctrl / Cmd toggles one. Pure, so the gallery grid and tests share it.
 */

export type Selection = { ids: ReadonlySet<string>; anchor: string | null }

export const EMPTY_SELECTION: Selection = { ids: new Set(), anchor: null }

export type SelectMode = "toggle" | "range" | "only"

export function select(selection: Selection, order: readonly string[], id: string, mode: SelectMode): Selection {
  if (mode === "only") return { ids: new Set([id]), anchor: id }
  if (mode === "range" && selection.anchor !== null) {
    const from = order.indexOf(selection.anchor)
    const to = order.indexOf(id)
    if (from !== -1 && to !== -1) {
      const [low, high] = from < to ? [from, to] : [to, from]
      const ids = new Set(selection.ids)
      for (const item of order.slice(low, high + 1)) ids.add(item)
      return { ids, anchor: selection.anchor }
    }
  }
  const ids = new Set(selection.ids)
  if (ids.has(id)) ids.delete(id)
  else ids.add(id)
  return { ids, anchor: id }
}

/** The mode a pointer / key event asks for. */
export function modeOf(event: { shiftKey: boolean; ctrlKey: boolean; metaKey: boolean }): SelectMode {
  if (event.shiftKey) return "range"
  return "toggle"
}

export function selectAll(order: readonly string[]): Selection {
  return { ids: new Set(order), anchor: order[0] ?? null }
}

/** Drops ids that are no longer listed (after a delete or a filter change). */
export function prune(selection: Selection, order: readonly string[]): Selection {
  const listed = new Set(order)
  const ids = new Set([...selection.ids].filter((id) => listed.has(id)))
  if (ids.size === selection.ids.size) return selection
  return { ids, anchor: selection.anchor && ids.has(selection.anchor) ? selection.anchor : null }
}

/** Selected ids in list order. */
export function ordered(selection: Selection, order: readonly string[]): string[] {
  return order.filter((id) => selection.ids.has(id))
}
