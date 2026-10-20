/**
 * "Send to" and "use these params": move a result (its image and/or its
 * generation parameters) into a form, switching to that form's page when
 * needed. Values are applied once the target form has loaded its schema, so
 * a page that mounts later still receives them (upload ids are not kept in
 * drafts, which a load would otherwise drop).
 */
import { clientOption } from "@/api/app-options"
import { api } from "@/api/endpoints"
import type { FormTab, ParamValues } from "@/api/types"
import { currentRoute, navigate, type Route, routeKey } from "@/app/router"
import { applyPaste, type PasteEntry, type PasteFilter, parsedEntries, planEntries } from "@/features/infotext/paste"
import type { FormIndex } from "@/ui/form-index"
import { getFormStore } from "@/ui/form-store"
import { describeFile } from "@/ui/uploads"

export type SendTarget = "txt2img" | "img2img" | "inpaint" | "extras"

/** The form a target fills. */
export const formOf = (target: SendTarget): FormTab => (target === "inpaint" ? "img2img" : target)

export function routeOf(tab: FormTab): Route {
  return tab === "extras" ? { name: "extras" } : { name: "generate", tab }
}

/** The core values that put an uploaded image into a target form (schema-format.md core names). */
export function imageEntry(target: SendTarget, fileId: string): PasteEntry | null {
  switch (target) {
    case "img2img":
      return { owner: "core", values: { init_img: fileId, mode: "img2img" } }
    case "inpaint":
      return { owner: "core", values: { init_img: fileId, mode: "Inpaint" } }
    case "extras":
      return { owner: "core", values: { image: fileId, mode: "Single Image" } }
    case "txt2img":
      return null
  }
}

/**
 * Image layers drawn over a source image (sketch, mask: image params whose
 * `source` names it) belong to that image: entries that set a new source
 * also clear its layers, unless they set them too.
 */
export function withLayersCleared(index: FormIndex, entries: readonly PasteEntry[]): PasteEntry[] {
  return entries.map((entry) => {
    const cleared: ParamValues = {}
    for (const [name, node] of index.inputs.get(entry.owner) ?? []) {
      if (node.type === "image" && node.source && node.source in entry.values && !(name in entry.values)) cleared[name] = null
    }
    return Object.keys(cleared).length > 0 ? { owner: entry.owner, values: { ...entry.values, ...cleared } } : entry
  })
}

type Delivery = { entries: PasteEntry[]; filter: PasteFilter; unsubscribe: () => void; resolve: (count: number) => void }
const pending = new Map<FormTab, Delivery>()

function apply(tab: FormTab, entries: readonly PasteEntry[], filter: PasteFilter): number {
  const store = getFormStore(tab)
  const { index } = store.getState()
  const plan = planEntries(index, withLayersCleared(index, entries), filter)
  return applyPaste({ store, tab }, plan.entries)
}

/**
 * Forms whose page keeps a loaded store across visits (the generate view
 * and Extras load a schema once and carry uploads over reloads). Other pages load
 * their schema again on every mount, which drops upload ids.
 */
const KEEPS_VALUES: ReadonlySet<FormTab> = new Set(["txt2img", "img2img", "extras"])

/**
 * Applies `entries` to form `tab` and shows its page: right away when the
 * form is loaded (and on screen, or a form that keeps its values),
 * otherwise after its next schema load. Resolves with the number of params set.
 */
export function deliver(tab: FormTab, entries: readonly PasteEntry[], filter: PasteFilter = {}): Promise<number> {
  const store = getFormStore(tab)
  const onScreen = routeKey(currentRoute()) === routeKey(routeOf(tab))
  if (store.getState().revision > 0 && (onScreen || KEEPS_VALUES.has(tab))) {
    pending.get(tab)?.unsubscribe()
    pending.delete(tab)
    const count = apply(tab, entries, filter)
    if (!onScreen) navigate(routeOf(tab))
    return Promise.resolve(count)
  }

  const previous = pending.get(tab)
  previous?.unsubscribe()
  previous?.resolve(0)
  return new Promise<number>((resolve) => {
    const unsubscribe = store.subscribe((s, prev) => {
      if (s.revision === prev.revision) return
      unsubscribe()
      pending.delete(tab)
      resolve(apply(tab, entries, filter))
    })
    pending.set(tab, { entries: [...entries], filter, unsubscribe, resolve })
    navigate(routeOf(tab))
  })
}

/** A result to send: its file, its parameters, and its pixel size when the output metadata has it. */
export type SendItem = { fileId?: string | null; infotext?: string | null; width?: number | null; height?: number | null }

/** The `send_seed` / `send_size` / `send_cfg` settings as a paste filter. */
export function sendFilter(): PasteFilter {
  return { seed: clientOption("send_seed"), size: clientOption("send_size"), cfg: clientOption("send_cfg") }
}

/**
 * Sends a result to a form: its image (output file ids are valid image
 * inputs) and, for the generation forms, its parameters (`send_seed` /
 * `send_size` / `send_cfg` respected).
 */
export async function sendTo(target: SendTarget, item: SendItem, filter: PasteFilter = sendFilter()): Promise<number> {
  const tab = formOf(target)
  const parsed = item.infotext && tab !== "extras" ? await api.parseInfotext(item.infotext, tab).catch(() => null) : null
  const entries = parsed ? parsedEntries(parsed) : []
  const image = item.fileId ? imageEntry(target, item.fileId) : null
  if (image && item.fileId) {
    entries.push(image)
    describeFile(item.fileId, item)
  }
  return deliver(tab, entries, filter)
}
