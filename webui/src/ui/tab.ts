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

/** The job body for the current values of a tab form. */
export function jobRequest(store: FormStore, schema: TabSchema): JobRequest {
  const { values } = store.getState()
  return {
    tab: schema.tab,
    core: values.core ?? {},
    scripts: Object.fromEntries(schema.scripts.map((script) => [script.id, values[script.id] ?? {}])),
  }
}
