/**
 * Pasted parameters set a unit's infotext fields (the server fills the ones
 * an entry leaves out with the unit defaults, as the old paste did), marked
 * `__paste__` (the server cannot know the unit they replace). Like the old
 * per-control paste, the rest of the unit (images, "Enable") is kept: a
 * store subscription merges the pasted fields into the previous value as
 * soon as the paste is written.
 */
import type { FormState, FormStore } from "@/ui/form-store"
import type { WidgetNode } from "@/ui/schema"
import { type Catalog, isPasted, mergePasted, normalizeUnit, type UnitProps } from "./unit"

export const WIDGET = "controlnet-unit"

const watched = new WeakSet<FormStore>()

function unitProps(node: unknown): Partial<UnitProps> | null {
  const widget = node as WidgetNode | undefined
  return widget?.type === "custom" && widget.widget === WIDGET ? ((widget.props ?? {}) as Partial<UnitProps>) : null
}

/**
 * The catalog the units of `owner` render from: a model refresh's, else the
 * one the schema ships once per tab, with the first unit (`unit_0`).
 */
export function catalogOf(state: FormState, owner: string, props: Partial<UnitProps>): Catalog | undefined {
  const refreshed = state.outputs[owner]?.catalog as Catalog | undefined
  return refreshed ?? props.catalog ?? unitProps(state.index.inputs.get(owner)?.get("unit_0"))?.catalog
}

/** Merges pasted units of `store` into the units they replace (once per store). */
export function watchPastes(store: FormStore): void {
  if (watched.has(store)) return
  watched.add(store)
  store.subscribe((state, prev) => {
    if (state.values === prev.values) return
    const merges: [owner: string, values: Record<string, unknown>][] = []
    for (const [owner, values] of Object.entries(state.values)) {
      if (!values || values === prev.values[owner]) continue
      const nodes = state.index.inputs.get(owner)
      const merged: Record<string, unknown> = {}
      for (const [name, value] of Object.entries(values)) {
        if (!isPasted(value) || value === prev.values[owner]?.[name]) continue
        const props = unitProps(nodes?.get(name))
        if (!props) continue
        const img2img = props.img2img === true
        merged[name] = mergePasted(normalizeUnit(prev.values[owner]?.[name], img2img), value, catalogOf(state, owner, props))
      }
      if (Object.keys(merged).length > 0) merges.push([owner, merged])
    }
    for (const [owner, values] of merges) store.getState().setValues(owner, values)
  })
}
