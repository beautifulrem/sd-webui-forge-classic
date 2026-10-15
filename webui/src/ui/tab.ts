/** Tab-form helpers: owners of a tab schema and the job body built from a form. */
import type { JobRequest } from "@/api/types"
import type { OwnerSpec } from "./form-index"
import type { FormStore } from "./form-store"
import type { TabSchema } from "./schema"

export function tabOwners(schema: TabSchema): OwnerSpec[] {
  return [
    { owner: "core", layout: schema.core.layout, defaults: schema.core.defaults ?? {} },
    ...schema.scripts.map((script) => ({ owner: script.id, layout: script.layout, defaults: script.defaults ?? {} })),
  ]
}

/** The job body for the current values of a tab form (without `submit: false` inputs). */
export function jobRequest(store: FormStore, schema: TabSchema): JobRequest {
  const { values, index } = store.getState()
  const submitted = (owner: string): Record<string, unknown> => {
    const ownerValues = values[owner] ?? {}
    const nodes = index.inputs.get(owner)
    if (!nodes || ![...nodes.values()].some((node) => node.submit === false)) return ownerValues
    return Object.fromEntries(Object.entries(ownerValues).filter(([name]) => nodes.get(name)?.submit !== false))
  }
  return {
    tab: schema.tab,
    core: submitted("core"),
    scripts: Object.fromEntries(schema.scripts.map((script) => [script.id, submitted(script.id)])),
  }
}
