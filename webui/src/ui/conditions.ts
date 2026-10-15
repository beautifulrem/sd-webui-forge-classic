/**
 * Condition evaluator for `visible_if` / `enabled_if` (schema-format.md §2).
 *
 * `param` addresses the same owner by name, another owner as
 * `"<owner>.<name>"` (`"core.<name>"` for the generation form). Missing
 * params resolve to `undefined`: falsy, `eq`/`in` false, `ne` true.
 * Comparisons follow Python like `remi/ui/conditions.py`: `True == 1`,
 * strings order by code point, mixed kinds never order (false).
 */
import type { Condition, Json } from "./schema"

/** Resolves a param reference (already split) to its current value. */
export type Resolver = (owner: string, name: string) => unknown

/** Splits `"script.param"` / `"param"` into `[owner, name]` relative to `self`. */
export function splitRef(ref: string, self: string): [owner: string, name: string] {
  const dot = ref.indexOf(".")
  return dot === -1 ? [self, ref] : [ref.slice(0, dot), ref.slice(dot + 1)]
}

/** Python-like truthiness, as the contract defines it. */
export function truthy(value: unknown): boolean {
  if (Array.isArray(value)) return value.length > 0
  return Boolean(value)
}

/**
 * Strict structural equality for JSON values (the "modified" indicator):
 * `true` and `1` differ, numbers and numeric strings do not match.
 */
export function jsonEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null) return false
  if (Array.isArray(a) !== Array.isArray(b)) return false
  if (Array.isArray(a)) {
    const other = b as unknown[]
    return a.length === other.length && a.every((item, i) => jsonEqual(item, other[i]))
  }
  const ak = Object.keys(a)
  const bk = Object.keys(b)
  return (
    ak.length === bk.length &&
    ak.every((key) => Object.hasOwn(b, key) && jsonEqual((a as Record<string, unknown>)[key], (b as Record<string, unknown>)[key]))
  )
}

/** A number for Python's numeric tower: numbers, and booleans as 0/1. */
function numeric(value: unknown): number | null {
  if (typeof value === "number") return value
  if (typeof value === "boolean") return Number(value)
  return null
}

/**
 * Python `==` on JSON values, as `remi/ui/conditions.py` evaluates them:
 * booleans equal the numbers 0/1 (`True == 1`), containers compare by content,
 * strings never equal numbers.
 */
export function pyEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true
  const na = numeric(a)
  const nb = numeric(b)
  if (na !== null || nb !== null) return na !== null && nb !== null && na === nb
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null) return false
  if (Array.isArray(a) !== Array.isArray(b)) return false
  if (Array.isArray(a)) {
    const other = b as unknown[]
    return a.length === other.length && a.every((item, i) => pyEqual(item, other[i]))
  }
  const ak = Object.keys(a)
  return (
    ak.length === Object.keys(b).length &&
    ak.every((key) => Object.hasOwn(b, key) && pyEqual((a as Record<string, unknown>)[key], (b as Record<string, unknown>)[key]))
  )
}

/** Code-point order of two strings (Python's, not UTF-16's). */
function compareStrings(a: string, b: string): number {
  const ca = Array.from(a, (c) => c.codePointAt(0) ?? 0)
  const cb = Array.from(b, (c) => c.codePointAt(0) ?? 0)
  for (let i = 0; i < Math.min(ca.length, cb.length); i++) {
    if (ca[i] !== cb[i]) return (ca[i] ?? 0) - (cb[i] ?? 0)
  }
  return ca.length - cb.length
}

/**
 * Python's ordering of two JSON values: negative / zero / positive, NaN for an
 * unordered pair (NaN operands), or null where Python raises `TypeError`
 * (mixed kinds, `None`, objects); the comparison is then false.
 */
function order(a: unknown, b: unknown): number | null {
  const na = numeric(a)
  const nb = numeric(b)
  if (na !== null && nb !== null) return na - nb
  if (typeof a === "string" && typeof b === "string") return compareStrings(a, b)
  if (Array.isArray(a) && Array.isArray(b)) {
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
      if (!pyEqual(a[i], b[i])) return order(a[i], b[i])
    }
    return a.length - b.length
  }
  return null
}

function compare(value: unknown, op: "gt" | "gte" | "lt" | "lte", operand: unknown): boolean {
  const result = order(value, operand)
  if (result === null) return false
  switch (op) {
    case "gt":
      return result > 0
    case "gte":
      return result >= 0
    case "lt":
      return result < 0
    case "lte":
      return result <= 0
  }
}

/**
 * Evaluates `cond` for `owner`. Absent conditions (undefined/null) are true.
 * Malformed conditions evaluate to true so a schema bug never hides controls.
 */
export function evaluate(cond: Condition | null | undefined, owner: string, resolve: Resolver): boolean {
  if (cond === undefined || cond === null) return true
  if (typeof cond === "boolean") return cond
  if (typeof cond !== "object") return true
  if ("all" in cond) return Array.isArray(cond.all) ? cond.all.every((c) => evaluate(c, owner, resolve)) : true
  if ("any" in cond) return Array.isArray(cond.any) ? cond.any.some((c) => evaluate(c, owner, resolve)) : true
  if ("not" in cond) return !evaluate(cond.not, owner, resolve)
  if (!("param" in cond) || typeof cond.param !== "string") return true

  const value = resolve(...splitRef(cond.param, owner))
  if ("eq" in cond) return value !== undefined && pyEqual(value, cond.eq)
  if ("ne" in cond) return value === undefined || !pyEqual(value, cond.ne)
  if ("in" in cond) return value !== undefined && Array.isArray(cond.in) && cond.in.some((v: Json) => pyEqual(value, v))
  for (const op of ["gt", "gte", "lt", "lte"] as const) {
    if (op in cond) return value !== undefined && compare(value, op, (cond as Record<string, unknown>)[op])
  }
  return truthy(value)
}

/** All `[owner, name]` pairs a condition reads (for dependency tracking / debugging). */
export function references(cond: Condition | null | undefined, owner: string): [string, string][] {
  if (!cond || typeof cond !== "object") return []
  if ("all" in cond) return cond.all.flatMap((c) => references(c, owner))
  if ("any" in cond) return cond.any.flatMap((c) => references(c, owner))
  if ("not" in cond) return references(cond.not, owner)
  return "param" in cond && typeof cond.param === "string" ? [splitRef(cond.param, owner)] : []
}
