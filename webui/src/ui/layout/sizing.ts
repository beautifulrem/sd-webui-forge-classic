import type { CSSProperties } from "react"
import type { Node } from "../schema"

/**
 * Sizing of a child inside a `row` (classes in index.css): `width` 1–12 is a
 * share of a 12-column row, "fill" takes the remaining space, "auto" (absent)
 * shares space evenly, and buttons hug their content. Spans and fills go full
 * width when the row itself is narrow (container query), so phone layouts stack.
 * Controls line up at the bottom of a row (a button next to a labelled input);
 * containers (columns, groups, sections, tabs) start at its top.
 */
const CONTAINERS = new Set(["column", "group", "section", "tabs"])

export function cellProps(node: Node): { className: string; style?: CSSProperties } {
  if (node.type === "button") return { className: "remi-cell remi-cell-content" }
  const top = CONTAINERS.has(node.type) ? " remi-cell-top" : ""
  const width = "width" in node ? node.width : undefined
  if (typeof width === "number" && width >= 1 && width <= 12) {
    return { className: `remi-cell remi-cell-span${top}`, style: { "--span": Math.round(width) } as CSSProperties }
  }
  if (width === "fill") return { className: `remi-cell remi-cell-fill${top}` }
  return { className: `remi-cell remi-cell-auto${top}` }
}
