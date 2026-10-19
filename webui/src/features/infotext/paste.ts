/**
 * Pasting parsed generation parameters (`POST /infotext/parse`, schema-format
 * §7) into a form: only params the form has are taken, optionally without
 * seed / size / CFG (the `send_seed` / `send_size` / `send_cfg` settings for "Send to"), and the
 * edited params' `on_change` actions and conflict locks run once afterwards.
 */
import type { InfotextParsed, ParamValues } from "@/api/types"
import { type ActionContext, CONFLICTS_ACTION, runAction } from "@/ui/actions"
import type { FormIndex } from "@/ui/form-index"

export type PasteEntry = { owner: string; values: ParamValues }

export type PasteFilter = {
  /** Take the seed params (Seed, Variation seed, …). */
  seed?: boolean
  /** Take the image size (Size-1 / Size-2). */
  size?: boolean
  /** Take the CFG scale. */
  cfg?: boolean
}

export type PastePlan = {
  entries: PasteEntry[]
  /** `"owner.param"` refs the form does not have. */
  skipped: string[]
}

const SEED_KEYS = new Set(["Seed", "Variation seed", "Variation seed strength", "Seed resize from-1", "Seed resize from-2"])
const SIZE_KEYS = new Set(["Size-1", "Size-2"])
const CFG_KEYS = new Set(["CFG scale"])

/** Parsed values as one entry per owner (core first). */
export function parsedEntries(parsed: Pick<InfotextParsed, "core" | "scripts">): PasteEntry[] {
  return [{ owner: "core", values: parsed.core ?? {} }, ...Object.entries(parsed.scripts ?? {}).map(([owner, values]) => ({ owner, values }))]
}

/** The values of `entries` that `index` (a loaded form) can take. */
export function planEntries(index: FormIndex, entries: readonly PasteEntry[], filter: PasteFilter = {}): PastePlan {
  const planned: PasteEntry[] = []
  const skipped: string[] = []
  for (const { owner, values } of entries) {
    const nodes = index.inputs.get(owner)
    const taken: ParamValues = {}
    for (const [name, value] of Object.entries(values ?? {})) {
      const node = nodes?.get(name)
      if (!node) {
        skipped.push(`${owner}.${name}`)
        continue
      }
      if (filter.seed === false && node.infotext && SEED_KEYS.has(node.infotext)) continue
      if (filter.size === false && node.infotext && SIZE_KEYS.has(node.infotext)) continue
      if (filter.cfg === false && node.infotext && CFG_KEYS.has(node.infotext)) continue
      taken[name] = value
    }
    if (Object.keys(taken).length > 0) planned.push({ owner, values: taken })
  }
  return { entries: planned, skipped }
}

/** The values of `parsed` that `index` (a loaded form) can take. */
export function pastePlan(index: FormIndex, parsed: Pick<InfotextParsed, "core" | "scripts">, filter: PasteFilter = {}): PastePlan {
  return planEntries(index, parsedEntries(parsed), filter)
}

/** Writes the plan into the form, then runs each distinct `on_change` once and one conflicts pass. */
export function applyPaste(ctx: ActionContext, entries: readonly PasteEntry[]): number {
  const state = ctx.store.getState()
  let count = 0
  let conflicts = false
  const actions: [owner: string, action: string][] = []
  for (const { owner, values } of entries) {
    state.setValues(owner, values)
    const nodes = state.index.inputs.get(owner)
    for (const name of Object.keys(values)) {
      count++
      const action = nodes?.get(name)?.on_change
      if (action && !actions.some(([o, a]) => o === owner && a === action)) actions.push([owner, action])
      if (state.conflictParams.has(`${owner}.${name}`)) conflicts = true
    }
  }
  for (const [owner, action] of actions) void runAction(ctx, owner, action)
  // A bulk change: no single "selected" param.
  if (conflicts) void runAction(ctx, "core", CONFLICTS_ACTION)
  return count
}
