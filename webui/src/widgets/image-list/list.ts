/** Pure edits of an ordered list (the widget's upload ids). */

/** `items` with the item at `from` moved to index `to` (clamped). */
export function moveItem<T>(items: readonly T[], from: number, to: number): T[] {
  const next = [...items]
  if (from < 0 || from >= next.length) return next
  const target = Math.min(Math.max(to, 0), next.length - 1)
  const [item] = next.splice(from, 1)
  next.splice(target, 0, item as T)
  return next
}

export function removeAt<T>(items: readonly T[], index: number): T[] {
  return items.filter((_, i) => i !== index)
}

export function replaceAt<T>(items: readonly T[], index: number, item: T): T[] {
  return items.map((existing, i) => (i === index ? item : existing))
}

/** The selection after removing `index` from a list of `length` items. */
export function selectionAfterRemove(index: number, length: number): number {
  return length <= 1 ? -1 : Math.min(index, length - 2)
}
