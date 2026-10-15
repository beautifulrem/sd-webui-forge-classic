/**
 * Keeps the run controllers in step with the server where WebSocket events
 * can be missed: after a reload (`restoreRuns` re-attaches the jobs the
 * previous page load left in flight) and after a reconnect or a queue change
 * that contradicts the local state (`wireRunSync`). Each job of a form is
 * looked up with `GET /jobs/{id}`; what finished meanwhile is settled and its
 * results filed into the session.
 *
 * Jobs are attached to their form *before* they are looked up, so events
 * that arrive while a lookup is pending apply directly; a lookup answer for a
 * job those events already settled is ignored.
 */
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import { useProgress } from "@/api/progress"
import type { Job } from "@/api/types"
import { onWsEvent } from "@/api/ws"
import { useSession } from "@/features/gallery/session-store"
import { dispatchRun, GENERATE_FORMS, type GenerateTab, persistedJobs, useRuns } from "./run-controller"
import type { JobOutcome } from "./run-machine"

let restoring: Promise<void> | null = null

/** Once per page load: re-attaches the persisted jobs, then checks them. */
export function restoreRuns(): Promise<void> {
  restoring ??= restore().catch(() => {})
  return restoring
}

async function restore() {
  for (const tab of GENERATE_FORMS) {
    const ids = persistedJobs(tab)
    if (ids.length === 0) continue
    for (const id of ids) useProgress.getState().track(id, tab)
    dispatchRun(tab, { type: "restore", jobs: ids, running: null })
  }
  await resyncRuns()
}

/** How a finished job counts for the run machine (null while it is queued or running). */
export function outcomeOf(status: Job["status"], interrupting: boolean): JobOutcome | null {
  switch (status) {
    case "queued":
    case "running":
      return null
    case "completed":
      return "completed"
    case "interrupted":
    case "cancelled":
      return "interrupted"
    case "failed":
      return interrupting ? "interrupted" : "failed"
  }
}

/** Applies the server's view of one job of `tab` (null: the server no longer knows it). */
function reconcile(tab: GenerateTab, jobId: string, job: Job | null) {
  const run = useRuns.getState().runs[tab]
  if (!run.jobs.includes(jobId)) return
  // A vanished job counts as failed: it ends forever / repeats instead of looping on nothing.
  const outcome = job ? outcomeOf(job.status, run.interrupting) : "failed"
  if (!outcome) {
    if (job?.status === "running" && run.running !== jobId) dispatchRun(tab, { type: "started", jobId })
    return
  }
  // The live progress of a job that has ended must not linger.
  if (useProgress.getState().running?.jobId === jobId) useProgress.getState().syncRunning(null)
  if (job && outcome !== "failed") useSession.getState().addJob({ jobId, tab, outputs: job.outputs, infotext: job.infotext })
  dispatchRun(tab, { type: "finished", jobId, outcome })
}

/** Job ids being looked up (a reconnect and a queue change can ask at once). */
const checking = new Set<string>()

async function check(tab: GenerateTab, jobId: string) {
  if (checking.has(jobId)) return
  checking.add(jobId)
  try {
    let job: Job | null
    try {
      job = await api.jobs.get(jobId)
    } catch (error) {
      // Gone (404); any other failure (server restarting) is retried on the next hello.
      if (!(error instanceof ApiError && error.status === 404)) return
      job = null
    }
    reconcile(tab, jobId, job)
  } finally {
    checking.delete(jobId)
  }
}

/** Looks up every job of `tabs` the run controllers wait for. */
export async function resyncRuns(tabs: readonly GenerateTab[] = GENERATE_FORMS): Promise<void> {
  const { runs } = useRuns.getState()
  await Promise.all(tabs.flatMap((tab) => runs[tab].jobs.map((jobId) => check(tab, jobId))))
}

let wired = false

/**
 * Resyncs after every `hello` (a reconnect: events sent while the socket was
 * down are lost) and when `queue.changed` names a running job other than the
 * one a form believes runs.
 */
export function wireRunSync() {
  if (wired) return
  wired = true
  onWsEvent("hello", () => void resyncRuns())
  onWsEvent("queue.changed", (e) => {
    if (e.running === undefined) return
    const running = e.running
    const { runs } = useRuns.getState()
    for (const tab of GENERATE_FORMS) {
      const run = runs[tab]
      // The job this form believes runs has ended without its event reaching us.
      if (run.running && run.running !== running) void resyncRuns([tab])
      if (running && run.jobs.includes(running) && run.running !== running) dispatchRun(tab, { type: "started", jobId: running })
    }
  })
}

/** Tests only. */
export function resetRestore() {
  restoring = null
  checking.clear()
}
