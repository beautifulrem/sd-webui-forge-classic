/**
 * One run controller per generation form: holds the `run-machine` state,
 * performs its effects (submit the form, interrupt, skip, cancel) and feeds
 * it the WebSocket job events. Jobs in flight are remembered in localStorage
 * so a reload can pick them up again (`restoreRuns`).
 */
import { toast } from "sonner"
import { create } from "zustand"
import { clientOption } from "@/api/app-options"
import { api } from "@/api/endpoints"
import { validationErrors } from "@/api/http"
import { useProgress } from "@/api/progress"
import { keys, queryClient } from "@/api/queries"
import type { FormTab, TabSchema } from "@/api/types"
import { onWsEvent } from "@/api/ws"
import { t } from "@/i18n"
import { getFormStore } from "@/ui/form-store"
import { jobRequest } from "@/ui/tab"
import { notifyResults } from "./effects"
import { IDLE, type JobOutcome, reduce, type RunEffect, type RunEvent, type RunState } from "./run-machine"

export type GenerateTab = Extract<FormTab, "txt2img" | "img2img">
export const GENERATE_FORMS: readonly GenerateTab[] = ["txt2img", "img2img"]

type RunStore = { runs: Record<GenerateTab, RunState> }

export const useRuns = create<RunStore>()(() => ({ runs: { txt2img: IDLE, img2img: IDLE } }))

export const persistKey = (tab: GenerateTab) => `remi-run:${tab}`

function persist(tab: GenerateTab, state: RunState) {
  try {
    if (state.jobs.length > 0) localStorage.setItem(persistKey(tab), JSON.stringify(state.jobs))
    else localStorage.removeItem(persistKey(tab))
  } catch {
    // Storage disabled: restoring after a reload is best-effort.
  }
}

/** Job ids a previous page load left in flight. */
export function persistedJobs(tab: GenerateTab): string[] {
  try {
    const raw = JSON.parse(localStorage.getItem(persistKey(tab)) ?? "[]") as unknown
    return Array.isArray(raw) ? raw.filter((id): id is string => typeof id === "string") : []
  } catch {
    return []
  }
}

export function dispatchRun(tab: GenerateTab, event: RunEvent) {
  const before = useRuns.getState().runs[tab]
  const [after, effects] = reduce(before, event, { interruptOnly: clientOption("ctrl_enter_interrupt") })
  if (after !== before) {
    useRuns.setState((s) => ({ runs: { ...s.runs, [tab]: after } }))
    if (after.jobs !== before.jobs) persist(tab, after)
  }
  for (const effect of effects) perform(tab, effect)
}

function perform(tab: GenerateTab, effect: RunEffect) {
  switch (effect.type) {
    case "submit":
      void submit(tab)
      return
    case "interrupt":
      void api.jobs.interrupt(effect.jobId).catch(() => {})
      return
    case "skip":
      void api.jobs.skip(effect.jobId).catch(() => {})
      return
    case "cancel":
      void api.jobs.remove(effect.jobId).catch(() => {})
      return
  }
}

async function submit(tab: GenerateTab) {
  const store = getFormStore(tab)
  const schema = queryClient.getQueryData<TabSchema>(keys.schema(tab))
  if (!schema) {
    dispatchRun(tab, { type: "rejected" })
    return
  }
  try {
    const created = await api.jobs.create(jobRequest(store, schema))
    store.getState().clearErrors()
    const jobIds = created.job_ids?.length ? created.job_ids : [created.job_id]
    for (const jobId of jobIds) useProgress.getState().track(jobId, tab)
    dispatchRun(tab, { type: "submitted", jobIds })
    for (const jobId of jobIds) replayEarly(tab, jobId)
  } catch (error) {
    const errors = validationErrors(error)
    if (errors) store.getState().setErrors(errors)
    toast.error(errors ? t("Some parameters are invalid") : t("Could not queue the job"), {
      description: error instanceof Error ? error.message : String(error),
    })
    dispatchRun(tab, { type: "rejected" })
  }
}

/**
 * Job events that arrived before the `POST /jobs` answer named their job
 * (a short job can start, even finish, first). Replayed once it is known.
 */
const early = new Map<string, { started?: boolean; outcome?: JobOutcome }>()
const EARLY_LIMIT = 50

function remember(jobId: string, event: { started?: boolean; outcome?: JobOutcome }) {
  early.set(jobId, { ...early.get(jobId), ...event })
  while (early.size > EARLY_LIMIT) early.delete(early.keys().next().value!)
}

function replayEarly(tab: GenerateTab, jobId: string) {
  const seen = early.get(jobId)
  if (!seen) return
  early.delete(jobId)
  if (seen.started) dispatchRun(tab, { type: "started", jobId })
  if (seen.outcome) dispatchRun(tab, { type: "finished", jobId, outcome: seen.outcome })
}

/** The form whose controller owns `jobId`, if any. */
function ownerOf(jobId: string): GenerateTab | null {
  const { runs } = useRuns.getState()
  return GENERATE_FORMS.find((tab) => runs[tab].jobs.includes(jobId)) ?? null
}

/** Whether one of the forms is waiting for a `POST /jobs` answer. */
const submitting = () => GENERATE_FORMS.some((tab) => useRuns.getState().runs[tab].pending > 0)

function finished(jobId: string, outcome: (tab: GenerateTab | null) => JobOutcome) {
  const tab = ownerOf(jobId)
  if (tab) dispatchRun(tab, { type: "finished", jobId, outcome: outcome(tab) })
  else if (submitting()) remember(jobId, { outcome: outcome(null) })
}

let wired = false

/** Subscribes the controllers to job events (once per app). */
export function wireRunControllers() {
  if (wired) return
  wired = true
  const started = (jobId: string) => {
    const tab = ownerOf(jobId)
    if (tab && useRuns.getState().runs[tab].running !== jobId) dispatchRun(tab, { type: "started", jobId })
    else if (!tab && submitting()) remember(jobId, { started: true })
  }
  onWsEvent("job.started", (e) => started(e.job_id))
  // A reload loses `job.started`: the first progress event marks the job running.
  onWsEvent("job.progress", (e) => started(e.job_id))
  onWsEvent("job.completed", (e) => {
    const stopped = e.status === "interrupted" || e.status === "cancelled"
    if (ownerOf(e.job_id) && !stopped) notifyResults(e.outputs.filter((o) => (o.kind ?? "image") === "image").length)
    finished(e.job_id, () => (stopped ? "interrupted" : "completed"))
  })
  onWsEvent("job.failed", (e) =>
    finished(e.job_id, (tab) =>
      e.status === "interrupted" || e.status === "cancelled" || (tab && useRuns.getState().runs[tab].interrupting) ? "interrupted" : "failed",
    ),
  )
}

/** Tests: back to idle, nothing persisted. */
export function resetRunControllers() {
  early.clear()
  useRuns.setState({ runs: { txt2img: IDLE, img2img: IDLE } })
  for (const tab of GENERATE_FORMS) localStorage.removeItem(persistKey(tab))
}
