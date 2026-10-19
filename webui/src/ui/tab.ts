/** Tab-form helpers: owners of a tab schema and the job body built from a form. */
import type { JobRequest } from "@/api/types"
import { evaluate } from "./conditions"
import type { OwnerSpec } from "./form-index"
import type { FormStore } from "./form-store"
import type { TabSchema, ValidationError } from "./schema"

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

/** No value: `null`/`undefined`, a blank string or an empty list (as the server sees it). */
function isEmpty(value: unknown): boolean {
  return value === null || value === undefined || (typeof value === "string" && !value.trim()) || (Array.isArray(value) && value.length === 0)
}

/**
 * The required core inputs (`core.required`) a job body leaves empty, as the
 * server's errors: checked before `POST /jobs` so a request that must fail
 * is never sent. A name the body leaves out takes its default, as on the server.
 */
export function missingInputs(schema: TabSchema, request: Pick<JobRequest, "core">): ValidationError[] {
  const core = request.core as Record<string, unknown>
  const defaults = (schema.core.defaults ?? {}) as Record<string, unknown>
  const value = (name: string) => (Object.hasOwn(core, name) ? core[name] : defaults[name])
  return (schema.core.required ?? [])
    .filter((entry) => evaluate(entry.when, "core", (owner, name) => (owner === "core" ? value(name) : undefined)) && isEmpty(value(entry.param)))
    .map((entry) => ({ owner: "core", param: entry.param, message: entry.message }))
}
