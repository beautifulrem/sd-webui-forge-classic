/**
 * Action client: POST /actions/{owner}/{action} with the owner's values and
 * apply the returned patch (schema-format.md §5).
 */
import { api } from "@/api/endpoints"
import { keys, queryClient } from "@/api/queries"
import type { ActionExtra, ActionRequest, ActionResponse, ActionToast, ParamPatch, ParamValues } from "@/api/types"
import { t } from "@/i18n"
import { showToast } from "@/lib/toast"
import { splitRef } from "./conditions"
import type { FormStore } from "./form-store"

const TOAST_LEVELS = new Set(["info", "success", "warning", "error"])
const PATCH_KEYS = new Set(["value", "choices", "visible", "enabled", "label", "info", "minimum", "maximum", "step"])

export type SplitResponse = {
  params: [owner: string, name: string, patch: ParamPatch][]
  toast?: ActionToast
  outputs?: Record<string, unknown>
  options?: Record<string, unknown>
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

/** Separates reserved keys from param patches and resolves `"owner.param"` keys. */
export function splitResponse(owner: string, response: ActionResponse | unknown): SplitResponse {
  const out: SplitResponse = { params: [] }
  if (!isRecord(response)) return out
  for (const [key, entry] of Object.entries(response)) {
    if (key === "__toast__") {
      if (isRecord(entry) && typeof entry.message === "string") {
        const level = TOAST_LEVELS.has(entry.level as string) ? (entry.level as ActionToast["level"]) : undefined
        out.toast = { level, message: entry.message }
      }
    } else if (key === "__outputs__") {
      if (isRecord(entry)) out.outputs = entry
    } else if (key === "__options__") {
      if (isRecord(entry)) out.options = entry
    } else if (key.startsWith("__")) {
      // Unknown reserved key: ignore (forward compatible).
    } else if (isRecord(entry)) {
      const patch: ParamPatch = {}
      for (const [field, value] of Object.entries(entry)) {
        if (PATCH_KEYS.has(field)) (patch as Record<string, unknown>)[field] = value
      }
      out.params.push([...splitRef(key, owner), patch])
    }
  }
  return out
}

export type ActionContext = {
  store: FormStore
  /** Form id sent as `tab`: txt2img / img2img / extras or `page:<id>`. */
  tab: string
}

/**
 * The request body: the owner's values, the core form's (for scripts) and
 * every other script's (`scripts`), so cross-owner actions such as
 * `core.__conflicts__` see the form as it is, not its defaults.
 */
export function actionRequest(ctx: ActionContext, owner: string, extra: ActionExtra = {}): ActionRequest {
  const { values } = ctx.store.getState()
  const scripts: Record<string, ParamValues> = {}
  for (const [id, ownerValues] of Object.entries(values)) {
    if (id !== "core" && id !== owner) scripts[id] = ownerValues
  }
  return {
    ...extra,
    tab: ctx.tab,
    values: values[owner] ?? {},
    ...(values.core && owner !== "core" ? { core: values.core } : {}),
    ...(Object.keys(scripts).length > 0 ? { scripts } : {}),
  }
}

const generations = new Map<string, number>()

/**
 * Runs `owner.action` and applies its patch. Responses that were overtaken by
 * a newer call of the same action are dropped (last write wins).
 */
export async function runAction(
  ctx: ActionContext,
  owner: string,
  action: string,
  extra: ActionExtra = {},
): Promise<SplitResponse | null> {
  const key = `${ctx.tab}/${owner}/${action}`
  const generation = (generations.get(key) ?? 0) + 1
  generations.set(key, generation)

  let response: ActionResponse
  try {
    response = await api.action(owner, action, actionRequest(ctx, owner, extra))
  } catch (error) {
    if (generations.get(key) === generation) {
      showToast("error", `${t("Action failed")}: ${error instanceof Error ? error.message : String(error)}`)
    }
    return null
  }
  if (generations.get(key) !== generation) return null

  const split = splitResponse(owner, response)
  applyResponse(ctx.store, owner, split)
  return split
}

export function applyResponse(store: FormStore, owner: string, split: SplitResponse) {
  const state = store.getState()
  if (split.params.length > 0) state.applyPatches(split.params)
  if (split.outputs) state.setOutputs(owner, split.outputs)
  if (split.toast) showToast(split.toast.level, split.toast.message)
  if (split.options) {
    void api
      .patchOptions(split.options)
      .then(() => queryClient.invalidateQueries({ queryKey: keys.app }))
      .catch((error: unknown) => showToast("error", `${t("Could not apply settings")}: ${String(error)}`))
  }
}

/** The core action resolving Anima conflict locks (schema-format.md §4, §5). */
export const CONFLICTS_ACTION = "__conflicts__"

/**
 * Runs what follows a user edit of `names` of `owner` (one field, or a bulk
 * reset): each distinct `on_change` action of those params, and the core
 * `__conflicts__` action when any of them is a conflict param. `selected` is
 * that reference when exactly one changed (the user's last action wins); a
 * bulk reset of several sends none.
 */
export function runChangeEffects(ctx: ActionContext, owner: string, names: readonly string[]) {
  const state = ctx.store.getState()
  const nodes = state.index.inputs.get(owner)
  const actions = new Set<string>()
  const conflicts: string[] = []
  for (const name of names) {
    const action = nodes?.get(name)?.on_change
    if (action) actions.add(action)
    if (state.conflictParams.has(`${owner}.${name}`)) conflicts.push(`${owner}.${name}`)
  }
  for (const action of actions) void runAction(ctx, owner, action)
  if (conflicts.length > 0) {
    void runAction(ctx, "core", CONFLICTS_ACTION, conflicts.length === 1 ? { selected: conflicts[0] } : {})
  }
}

/** Resets `names` of `owner` (all when omitted) to their defaults and runs the change effects. */
export function resetParams(ctx: ActionContext, owner: string, names?: readonly string[]) {
  const before = ctx.store.getState().values[owner] ?? {}
  ctx.store.getState().reset(owner, names)
  const after = ctx.store.getState().values[owner] ?? {}
  const changed = Object.keys(after).filter((name) => (names ? names.includes(name) : true) && after[name] !== before[name])
  if (changed.length > 0) runChangeEffects(ctx, owner, changed)
}
