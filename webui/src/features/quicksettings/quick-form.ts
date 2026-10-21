/**
 * The quick settings form: the pinned settings in their own form store,
 * written to the server as soon as they change (continuous kinds debounced),
 * like the Gradio quicksettings row. A rejected value goes back to the stored
 * one and shows the server's message under the control.
 */
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import { validationErrors } from "@/api/http"
import { keys, queryClient } from "@/api/queries"
import { t } from "@/i18n"
import { jsonEqual } from "@/ui/conditions"
import { createFormStore, type FormStore } from "@/ui/form-store"
import type { GroupNode, SettingNode, SettingsSchema } from "@/ui/schema"

export const QUICK_OWNER = "settings"
const CONTINUOUS = new Set(["slider", "number", "text", "color", "json"])
export const WRITE_DELAY = 400

export type QuickForm = {
  store: FormStore
  nodes: SettingNode[]
  /** Loads `schema`'s values for `keys` (keeps edits being written). */
  load: (schema: SettingsSchema, keys: readonly string[]) => void
  /** Starts writing changes; the returned function stops (flushing a pending write). */
  start: () => () => void
}

type Writer = (values: Record<string, unknown>) => Promise<unknown>

function inputsByKey(schema: SettingsSchema): Map<string, SettingNode> {
  const out = new Map<string, SettingNode>()
  const visit = (node: SettingsSchema["categories"][number]["sections"][number]["items"][number]) => {
    if (node.type === "row") for (const child of node.children) visit(child as never)
    else if (node.type !== "html" && node.type !== "markdown" && node.type !== "separator") out.set(node.name, node)
  }
  for (const category of schema.categories) for (const section of category.sections) section.items.forEach(visit)
  return out
}

export function createQuickForm(write: Writer = (values) => api.patchOptions(values), delay = WRITE_DELAY): QuickForm {
  const store = createFormStore("quicksettings", { persistDrafts: false })
  let saved: Record<string, unknown> = {}
  let nodes: SettingNode[] = []
  let timer: ReturnType<typeof setTimeout> | undefined
  let loading = false

  const flush = async () => {
    clearTimeout(timer)
    timer = undefined
    const values = store.getState().values[QUICK_OWNER] ?? {}
    const changes = Object.fromEntries(nodes.filter((node) => !jsonEqual(values[node.name], saved[node.name])).map((node) => [node.name, values[node.name]]))
    if (Object.keys(changes).length === 0) return
    const previous = saved
    saved = { ...saved, ...changes }
    try {
      await write(changes)
      void queryClient.invalidateQueries({ queryKey: keys.options })
      void queryClient.invalidateQueries({ queryKey: keys.app })
    } catch (error) {
      saved = previous
      const reverted = Object.fromEntries(Object.keys(changes).map((key) => [key, previous[key]]))
      loading = true
      store.getState().setValues(QUICK_OWNER, reverted)
      loading = false
      const errors = validationErrors(error)
      if (errors) store.getState().setErrors(errors.map((e) => ({ ...e, owner: QUICK_OWNER })))
      toast.error(t("Could not change the setting"), { description: error instanceof Error ? error.message : String(error) })
    }
  }

  const listener = (state: ReturnType<FormStore["getState"]>, before: ReturnType<FormStore["getState"]>) => {
    if (loading || state.values === before.values) return
    const now = state.values[QUICK_OWNER] ?? {}
    const then = before.values[QUICK_OWNER] ?? {}
    const changed = Object.keys(now).filter((key) => !jsonEqual(now[key], then[key]))
    if (changed.length === 0) return
    const continuous = changed.some((key) => CONTINUOUS.has(nodes.find((node) => node.name === key)?.type ?? ""))
    clearTimeout(timer)
    if (continuous) timer = setTimeout(() => void flush(), delay)
    else void flush()
  }

  return {
    store,
    get nodes() {
      return nodes
    },
    load(schema, keyList) {
      const byKey = inputsByKey(schema)
      nodes = keyList.map((key) => byKey.get(key)).filter((node): node is SettingNode => node !== undefined && node.type !== "state")
      const layout: GroupNode = { type: "group", children: nodes }
      const pending = timer !== undefined ? (store.getState().values[QUICK_OWNER] ?? {}) : {}
      loading = true
      const defaults = { ...schema.values, ...schema.defaults }
      store.getState().load([{ owner: QUICK_OWNER, layout, defaults: Object.fromEntries(nodes.map((node) => [node.name, defaults[node.name] ?? node.default ?? null])) }])
      const values = Object.fromEntries(nodes.map((node) => [node.name, Object.hasOwn(pending, node.name) && !jsonEqual(pending[node.name], saved[node.name]) ? pending[node.name] : schema.values[node.name]]))
      store.getState().setValues(QUICK_OWNER, values)
      loading = false
      saved = { ...saved, ...Object.fromEntries(nodes.map((node) => [node.name, schema.values[node.name]])) }
    },
    start() {
      const unsubscribe = store.subscribe(listener)
      return () => {
        unsubscribe()
        if (timer !== undefined) void flush()
      }
    },
  }
}
