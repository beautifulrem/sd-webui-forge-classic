/**
 * Live generation state derived from WebSocket job events, fed by
 * `app/ws-bridge.ts`: the running job's progress (global: one worker), the
 * results of the jobs this client queued (per form, so txt2img results do
 * not show on img2img and other clients' jobs never replace ours), and the
 * model-loading state.
 */
import { create } from "zustand"
import type { WsEventMap } from "./events"
import type { FormTab, JobOutput } from "./types"

export type RunningJob = {
  jobId: string
  seq?: number
  step: number
  steps: number
  eta?: number | null
  jobNo?: number
  jobCount?: number
  text?: string
}

export type FinishedJob = {
  jobId: string
  outputs: JobOutput[]
  infotext?: string | null
  error?: string
}

/** How many owned / finished jobs are remembered. */
const KEEP = 50

type ProgressState = {
  running: RunningJob | null
  /** Jobs queued by this client, with their form (insertion order = queue order). */
  owned: Record<string, FormTab>
  /** The newest finished job per form (only jobs this client queued). */
  last: Partial<Record<FormTab, FinishedJob>>
  /** Finished jobs whose POST /jobs answer had not arrived yet (tiny jobs can finish first). */
  orphans: Record<string, FinishedJob>
  modelLoading: WsEventMap["model.loading"] | null
  /** Records a job queued by this client (from the `POST /jobs` or retry answer). */
  track: (jobId: string, tab: FormTab) => void
  started: (e: WsEventMap["job.started"]) => void
  progress: (e: WsEventMap["job.progress"]) => void
  completed: (e: WsEventMap["job.completed"]) => void
  failed: (e: WsEventMap["job.failed"]) => void
  setModelLoading: (e: WsEventMap["model.loading"] | null) => void
  /** The server runs `jobId` (null: nothing): forgets a running job it no longer runs (missed events). */
  syncRunning: (jobId: string | null) => void
}

function capped<T>(record: Record<string, T>): Record<string, T> {
  const keys = Object.keys(record)
  return keys.length <= KEEP ? record : Object.fromEntries(keys.slice(-KEEP).map((key) => [key, record[key] as T]))
}

export const useProgress = create<ProgressState>()((set) => {
  function finish(job: FinishedJob) {
    set((s) => {
      const running = s.running?.jobId === job.jobId ? null : s.running
      const tab = s.owned[job.jobId]
      if (!tab) return { running, orphans: capped({ ...s.orphans, [job.jobId]: job }) }
      return { running, last: { ...s.last, [tab]: job } }
    })
  }

  return {
    running: null,
    owned: {},
    last: {},
    orphans: {},
    modelLoading: null,
    track: (jobId, tab) =>
      set((s) => {
        const orphan = s.orphans[jobId]
        if (!orphan) return { owned: capped({ ...s.owned, [jobId]: tab }) }
        const orphans = { ...s.orphans }
        delete orphans[jobId]
        return { owned: capped({ ...s.owned, [jobId]: tab }), orphans, last: { ...s.last, [tab]: orphan } }
      }),
    started: (e) => set({ running: { jobId: e.job_id, seq: e.seq, step: 0, steps: e.steps ?? 0, jobCount: e.job_count } }),
    progress: (e) =>
      set({
        running: {
          jobId: e.job_id,
          seq: e.seq,
          step: e.step,
          steps: e.steps,
          eta: e.eta,
          jobNo: e.job_no,
          jobCount: e.job_count,
          text: e.text,
        },
      }),
    completed: (e) => finish({ jobId: e.job_id, outputs: e.outputs ?? [], infotext: e.infotext }),
    failed: (e) => finish({ jobId: e.job_id, outputs: [], error: e.error }),
    setModelLoading: (modelLoading) => set({ modelLoading }),
    syncRunning: (jobId) => set((s) => (s.running && s.running.jobId !== jobId ? { running: null } : s)),
  }
})

type ProgressSnapshot = Pick<ProgressState, "running" | "owned" | "last">

/** The newest result of a form's jobs (queued by this client). */
export function lastResult(s: ProgressSnapshot, tab: string): FinishedJob | undefined {
  return s.last[tab as FormTab]
}

/** The running job when this client queued it from `tab`. */
export function runningFor(s: ProgressSnapshot, tab: string): RunningJob | null {
  return s.running && s.owned[s.running.jobId] === tab ? s.running : null
}

/** 0..1 fraction of the running job, or null when idle. */
export function progressFraction(job: RunningJob | null): number | null {
  if (!job) return null
  if (!job.steps) return 0
  const perJob = Math.min(1, job.step / job.steps)
  if (job.jobCount && job.jobCount > 1 && job.jobNo !== undefined) {
    return Math.min(1, (job.jobNo + perJob) / job.jobCount)
  }
  return perJob
}
