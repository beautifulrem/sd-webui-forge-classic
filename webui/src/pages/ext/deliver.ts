/**
 * Values sent to an extension page from elsewhere (PNG Info "Send to
 * JoyCaption", a page's own send buttons). A page's form (`page:<id>`) keeps
 * its values across visits, so a loaded page takes them at once; one that
 * has not loaded its schema yet takes them right after its first load. A
 * send to a page that is missing or fails to load settles with 0.
 */
import { hashKey } from "@tanstack/react-query"
import { keys, queryClient } from "@/api/queries"
import type { ParamValues } from "@/api/types"
import { navigate, type Route } from "@/app/router"
import { runChangeEffects } from "@/ui/actions"
import { type FormStore, getFormStore } from "@/ui/form-store"
import type { PageSchema } from "@/ui/schema"

export const pageOwner = (id: string) => `page:${id}`
export const pageRoute = (id: string): Extract<Route, { name: "page" }> => ({ name: "page", id })
export const pageStore = (id: string): FormStore => getFormStore(pageOwner(id))

/** The schema object each page store was last loaded with. */
const loadedWith = new WeakMap<FormStore, PageSchema>()

export function loadedSchema(store: FormStore): PageSchema | undefined {
  return loadedWith.get(store)
}

/**
 * Loads `schema` into its page store unless that exact schema is already
 * there; current values (uploads included) and outputs carry over a reload.
 */
export function loadPage(schema: PageSchema): FormStore {
  const store = pageStore(schema.id)
  if (loadedWith.get(store) === schema) return store
  const owner = schema.owner || pageOwner(schema.id)
  const reloading = loadedWith.has(store)
  const { outputs, values } = store.getState()
  // Recorded first: views re-render (and read it) while the store loads
  loadedWith.set(store, schema)
  store.getState().load([{ owner, layout: schema.layout, defaults: schema.defaults ?? {} }])
  if (reloading) {
    // A load never restores upload ids (drafts); within the session they are still valid.
    const uploads = Object.fromEntries(
      [...(store.getState().index.inputs.get(owner) ?? [])]
        .filter(([name, node]) => UPLOAD_KINDS.has(node.type) && values[owner]?.[name] != null)
        .map(([name]) => [name, values[owner]![name]]),
    )
    if (Object.keys(uploads).length > 0) store.getState().setValues(owner, uploads)
  }
  if (outputs[owner]) store.getState().setOutputs(owner, outputs[owner])
  return store
}

const UPLOAD_KINDS = new Set(["image", "image_list", "file"])

/** Sets the values the page declares (others are dropped) and runs their `on_change` actions. */
export function applyToPage(id: string, values: ParamValues): number {
  const store = pageStore(id)
  const owner = pageOwner(id)
  const known = store.getState().index.inputs.get(owner)
  const accepted = Object.fromEntries(Object.entries(values).filter(([name]) => known?.has(name)))
  const names = Object.keys(accepted)
  if (names.length === 0) return 0
  store.getState().setValues(owner, accepted)
  runChangeEffects({ store, tab: owner }, owner, names)
  return names.length
}

/** A send waiting for its page longer than this gives up (resolves 0). */
export const PENDING_TIMEOUT_MS = 30_000

type Pending = { values: ParamValues; resolve: (count: number) => void; stop: () => void }
const pending = new Map<string, Pending>()

function settle(id: string, count: (entry: Pending) => number): void {
  const entry = pending.get(id)
  if (!entry) return
  pending.delete(id)
  entry.stop()
  entry.resolve(count(entry))
}

/** Puts `values` into page `id` and shows it; resolves with the number of params set. */
export function sendToPage(id: string, values: ParamValues): Promise<number> {
  dropPending(id)
  if (loadedWith.has(pageStore(id))) {
    const count = applyToPage(id, values)
    navigate(pageRoute(id))
    return Promise.resolve(count)
  }
  return new Promise<number>((resolve) => {
    // The page view loads the schema; a failed load ends the wait.
    const hash = hashKey(keys.page(id))
    const unsubscribe = queryClient.getQueryCache().subscribe((event) => {
      if (event.query.queryHash === hash && event.query.state.status === "error") dropPending(id)
    })
    const timer = setTimeout(() => dropPending(id), PENDING_TIMEOUT_MS)
    pending.set(id, {
      values,
      resolve,
      stop: () => {
        unsubscribe()
        clearTimeout(timer)
      },
    })
    navigate(pageRoute(id))
  })
}

/** Called by the page view once its schema is loaded: applies a pending send. */
export function flushPending(id: string): void {
  settle(id, (entry) => applyToPage(id, entry.values))
}

/** Gives up a pending send to page `id` (missing, failed to load): it resolves with 0. */
export function dropPending(id: string): void {
  settle(id, () => 0)
}
