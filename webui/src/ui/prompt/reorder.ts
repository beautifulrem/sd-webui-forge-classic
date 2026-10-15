/**
 * Alt+←/→ moves the comma-separated tag(s) under the selection one position
 * left or right (port of javascript/edit-order.js, as a pure function).
 */
import type { Edit } from "./attention"

export function moveTags(text: string, from: number, to: number, direction: -1 | 1): Edit | null {
  const start = Math.min(from, to)
  const end = Math.max(from, to)
  const items = text.split(",")
  const indexStart = (text.slice(0, start).match(/,/g) ?? []).length
  const indexEnd = (text.slice(0, end).match(/,/g) ?? []).length
  const range = indexEnd - indexStart + 1

  if (direction === -1 && indexStart > 0) {
    items.splice(indexStart - 1, 0, ...items.splice(indexStart, range))
    return {
      text: items.join(","),
      from: items.slice(0, indexStart - 1).join(",").length + (indexStart === 1 ? 0 : 1),
      to: items.slice(0, indexEnd).join(",").length,
    }
  }
  if (direction === 1 && indexEnd < items.length - 1) {
    items.splice(indexStart + 1, 0, ...items.splice(indexStart, range))
    return {
      text: items.join(","),
      from: items.slice(0, indexStart + 1).join(",").length + 1,
      to: items.slice(0, indexEnd + 2).join(",").length,
    }
  }
  return null
}
