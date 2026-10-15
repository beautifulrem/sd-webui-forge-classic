/** Client-side value rules per input kind (mirrors schema-format.md §3, leniently). */
import type { InputNode } from "./schema"

const NOT_ACCEPTED = Symbol("not accepted")

export function clamp(value: number, min?: number | null, max?: number | null): number {
  if (min !== undefined && min !== null && value < min) return min
  if (max !== undefined && max !== null && value > max) return max
  return value
}

/** Rounds to the step grid anchored at `base`, avoiding float noise (0.1 + 0.2). */
export function snapToStep(value: number, step: number | null | undefined, base = 0): number {
  if (!step || step <= 0) return value
  const snapped = Math.round((value - base) / step) * step + base
  const decimals = (String(step).split(".")[1] ?? "").length
  return Number(snapped.toFixed(Math.min(decimals + 2, 12)))
}

/** Decimal places implied by a step (0.05 → 2), for display. */
export function stepDecimals(step: number | null | undefined): number {
  if (!step) return 0
  const text = String(step)
  if (text.includes("e-")) return Number(text.split("e-")[1])
  return (text.split(".")[1] ?? "").length
}

/** Rounds `value` to `decimals` places without float noise. */
export function roundTo(value: number, decimals: number): number {
  return Number(value.toFixed(Math.max(0, Math.min(decimals, 100))))
}

/**
 * The server's coercion of a slider / number (§3): clamped to the bounds,
 * rounded when integer, and to `precision` places for numbers.
 */
export function coerceNumber(
  node: Extract<InputNode, { type: "slider" | "number" }>,
  value: number,
  /** Bounds after action overrides. */
  bounds: { minimum?: number | null; maximum?: number | null } = node,
): number {
  let next = clamp(value, bounds.minimum, bounds.maximum)
  if (node.integer) next = Math.round(next)
  else if (node.type === "number" && typeof node.precision === "number") next = roundTo(next, node.precision)
  return next
}

const HEX_COLOR = /^#(?:[0-9a-f]{3}|[0-9a-f]{6}|[0-9a-f]{8})$/i

export function isHexColor(value: unknown): value is string {
  return typeof value === "string" && HEX_COLOR.test(value)
}

/**
 * Validates a persisted draft value against the current node. Returns the
 * (possibly clamped) value, or `NOT_ACCEPTED` when the default should win.
 */
function acceptDraft(node: InputNode, value: unknown): unknown {
  switch (node.type) {
    case "toggle":
      return typeof value === "boolean" ? value : NOT_ACCEPTED
    case "slider":
    case "number":
      return typeof value === "number" && Number.isFinite(value) ? coerceNumber(node, value) : NOT_ACCEPTED
    case "seed":
      return typeof value === "number" && Number.isInteger(value) ? value : NOT_ACCEPTED
    case "text":
      return typeof value === "string" ? value : NOT_ACCEPTED
    case "color":
      return isHexColor(value) ? value : NOT_ACCEPTED
    case "choice":
      if (node.choices.some((c) => c.value === value)) return value
      return node.allow_custom && typeof value === "string" ? value : NOT_ACCEPTED
    case "multichoice": {
      if (!Array.isArray(value)) return NOT_ACCEPTED
      return node.allow_custom ? value : value.filter((v) => node.choices.some((c) => c.value === v))
    }
    // Upload ids are short-lived on the server; never restore them from drafts.
    case "image":
    case "image_list":
    case "file":
      return NOT_ACCEPTED
    case "json":
    case "custom":
      return value === undefined ? NOT_ACCEPTED : value
    // Hidden server state belongs to the session that produced it.
    case "state":
      return NOT_ACCEPTED
  }
}

/** The value a freshly loaded form should hold for `node`. */
export function restoreValue(node: InputNode, draft: unknown, fallback: unknown): unknown {
  if (draft === undefined) return fallback
  const accepted = acceptDraft(node, draft)
  return accepted === NOT_ACCEPTED ? fallback : accepted
}

/** Whether drafts of this kind are persisted at all (not uploads, not hidden server state). */
export function isPersisted(node: InputNode): boolean {
  return node.type !== "image" && node.type !== "image_list" && node.type !== "file" && node.type !== "state"
}
