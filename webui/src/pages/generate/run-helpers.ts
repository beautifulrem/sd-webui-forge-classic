/** Run-control helpers: queue position, form reset. */
import type { Job } from "@/api/types"
import type { GenerateTab } from "@/features/progress/run-controller"
import { resetOwners } from "@/ui/actions"
import { getFormStore } from "@/ui/form-store"

/**
 * Jobs ahead of `jobId`: its place in `queued` (`GET /queue/items`, already in
 * run order) plus the running job. Null when it is not queued.
 */
export function jobsAhead(queued: readonly Job[], jobId: string, running: boolean): number | null {
  const index = queued.findIndex((job) => job.id === jobId)
  if (index === -1) return null
  return index + (running ? 1 : 0)
}

const PROMPTS = ["prompt", "negative_prompt"]

const filled = (value: unknown) =>
  Array.isArray(value) ? value.length > 0 : typeof value === "string" ? value.trim() !== "" : value !== null && value !== undefined

/** Whether resetting the form would throw away prompt text or source images. */
export function resetLosesWork(tab: GenerateTab): boolean {
  const { values, index } = getFormStore(tab).getState()
  if (PROMPTS.some((name) => filled(values.core?.[name]))) return true
  for (const [owner, inputs] of index.inputs) {
    for (const [name, node] of inputs) {
      if ((node.type === "image" || node.type === "image_list") && filled(values[owner]?.[name])) return true
    }
  }
  return false
}

/** Every owner back to its defaults, with one round of change effects (one conflicts call). */
export function resetForm(tab: GenerateTab) {
  const store = getFormStore(tab)
  resetOwners({ store, tab }, Object.keys(store.getState().defaults))
}
