/**
 * Generate / Interrupt / Skip state machine of one form, as a pure reducer:
 * `(state, event) → [state, effects]`. The controller (`run-controller.ts`)
 * feeds it user input and WebSocket job events and performs the effects.
 *
 * - Generate queues a job; pressing it again while one runs queues another.
 * - Ctrl/Cmd+Enter while this form's job runs interrupts it and, unless the
 *   setting `ctrl_enter_interrupt` is on, generates again once it has ended.
 * - Interrupt stops the running job, cancels this form's queued jobs and
 *   ends "generate forever" and repeats. Skip only skips the current image.
 * - "Generate forever" and "repeat N" queue the next job when the previous
 *   one has finished (so each run picks up edits); a failure stops them. A
 *   reload keeps them going (`resume`, after the jobs are restored).
 */

export type RunPhase = "idle" | "submitting" | "queued" | "running" | "interrupting"

export type RunState = {
  /** Jobs this form queued that have not finished, in queue order. */
  jobs: readonly string[]
  /** The job of `jobs` the worker is running. */
  running: string | null
  /** `POST /jobs` requests in flight (Generate can be pressed again meanwhile). */
  pending: number
  interrupting: boolean
  forever: boolean
  /** Runs still to queue, one after another (repeat N). */
  repeatLeft: number
  /** Generate again when the interrupted job has ended (Ctrl/Cmd+Enter). */
  restart: boolean
  /** In-flight submits to cancel when they answer (interrupted while they were pending). */
  dropPending: number
}

export type JobOutcome = "completed" | "failed" | "interrupted"

export type RunEvent =
  /** `runs` > 1: generate that many times in a row. */
  | { type: "generate"; runs?: number }
  | { type: "shortcut" }
  /** A `POST /jobs` answer: the jobs it queued (several with `repeat`). */
  | { type: "submitted"; jobIds: readonly string[] }
  | { type: "rejected" }
  | { type: "started"; jobId: string }
  | { type: "finished"; jobId: string; outcome: JobOutcome }
  | { type: "interrupt" }
  | { type: "skip" }
  | { type: "forever"; on: boolean }
  /** Drop the remaining repeats (the current run goes on). */
  | { type: "stop-repeats" }
  /** Jobs found again after a page reload. */
  | { type: "restore"; jobs: readonly string[]; running: string | null }
  /**
   * The loops ("Generate forever", runs left of "Generate several times") a
   * previous page load had going: they go on after the restored jobs, if one
   * of them is still in flight.
   */
  | { type: "resume"; forever: boolean; repeatLeft: number }
  /** Hires. fix an existing result with the form's values (txt2img's ✨ button): one job. */
  | { type: "upscale"; source: UpscaleSource }

/** The result a Hires. fix-only run starts from. */
export type UpscaleSource = { fileId: string; key: string }

export type RunEffect =
  | { type: "submit"; upscale?: UpscaleSource }
  | { type: "interrupt"; jobId: string }
  | { type: "skip"; jobId: string }
  /** Remove a queued (not yet started) job. */
  | { type: "cancel"; jobId: string }

export type RunOptions = {
  /** `ctrl_enter_interrupt`: Ctrl/Cmd+Enter only interrupts. */
  interruptOnly: boolean
}

export const IDLE: RunState = {
  jobs: [],
  running: null,
  pending: 0,
  interrupting: false,
  forever: false,
  repeatLeft: 0,
  restart: false,
  dropPending: 0,
}

export function phaseOf(s: RunState): RunPhase {
  if (s.interrupting) return "interrupting"
  if (s.running) return "running"
  if (s.jobs.length > 0) return "queued"
  if (s.pending > 0) return "submitting"
  return "idle"
}

/** Busy = something of this form is queued, running or being submitted. */
export const isBusy = (s: RunState) => s.pending > 0 || s.jobs.length > 0

type Result = [RunState, RunEffect[]]

function submit(s: RunState): Result {
  return [{ ...s, pending: s.pending + 1 }, [{ type: "submit" }]]
}

function interrupt(s: RunState): Result {
  const effects: RunEffect[] = []
  if (s.running) effects.push({ type: "interrupt", jobId: s.running })
  const queued = s.jobs.filter((id) => id !== s.running)
  for (const jobId of queued) effects.push({ type: "cancel", jobId })
  return [
    {
      ...s,
      jobs: s.running ? [s.running] : [],
      interrupting: Boolean(s.running),
      forever: false,
      repeatLeft: 0,
      restart: false,
      dropPending: s.pending,
    },
    effects,
  ]
}

/** What follows when this form has nothing queued any more. */
function settle(s: RunState, outcome: JobOutcome, effects: RunEffect[] = []): Result {
  if (s.jobs.length > 0 || s.pending > 0) return [s, effects]
  const next = { ...s, interrupting: false }
  const then = ([state, more]: Result): Result => [state, [...effects, ...more]]
  if (next.restart) return then(submit({ ...next, restart: false }))
  if (outcome === "failed") return [{ ...next, forever: false, repeatLeft: 0 }, effects]
  if (next.forever) return then(submit(next))
  if (next.repeatLeft > 0) return then(submit({ ...next, repeatLeft: next.repeatLeft - 1 }))
  return [next, effects]
}

export function reduce(s: RunState, event: RunEvent, options: RunOptions): Result {
  switch (event.type) {
    case "generate": {
      const runs = Math.max(1, Math.floor(event.runs ?? 1))
      return submit(runs > 1 ? { ...s, repeatLeft: runs - 1 } : s)
    }
    case "shortcut": {
      // A second press while the first submit is in flight must not queue a duplicate.
      if (!s.running) return s.pending > 0 ? [s, []] : submit(s)
      const [next, effects] = interrupt(s)
      return [{ ...next, restart: !options.interruptOnly }, effects]
    }
    case "submitted": {
      const pending = Math.max(0, s.pending - 1)
      if (s.dropPending > 0) {
        const effects = event.jobIds.map<RunEffect>((jobId) => ({ type: "cancel", jobId }))
        return settle({ ...s, pending, dropPending: s.dropPending - 1 }, "interrupted", effects)
      }
      const added = event.jobIds.filter((id) => !s.jobs.includes(id))
      return [{ ...s, pending, jobs: added.length > 0 ? [...s.jobs, ...added] : s.jobs }, []]
    }
    case "rejected": {
      const next = { ...s, pending: Math.max(0, s.pending - 1), dropPending: Math.max(0, s.dropPending - 1) }
      return [{ ...next, forever: false, repeatLeft: 0, restart: false }, []]
    }
    case "started":
      return s.jobs.includes(event.jobId) ? [{ ...s, running: event.jobId }, []] : [s, []]
    case "finished": {
      if (!s.jobs.includes(event.jobId)) return [s, []]
      const next = { ...s, jobs: s.jobs.filter((id) => id !== event.jobId), running: s.running === event.jobId ? null : s.running }
      return settle(next, event.outcome)
    }
    case "interrupt":
      return interrupt(s)
    case "skip":
      return [s, s.running ? [{ type: "skip", jobId: s.running }] : []]
    case "forever":
      if (!event.on) return [{ ...s, forever: false }, []]
      return isBusy(s) ? [{ ...s, forever: true }, []] : submit({ ...s, forever: true })
    case "stop-repeats":
      return [{ ...s, repeatLeft: 0 }, []]
    case "upscale":
      return [{ ...s, pending: s.pending + 1 }, [{ type: "submit", upscale: event.source }]]
    case "resume":
      // Nothing in flight any more: the loop ended while the page was away.
      if (!isBusy(s)) return [s, []]
      return [{ ...s, forever: s.forever || event.forever, repeatLeft: Math.max(s.repeatLeft, event.repeatLeft) }, []]
    case "restore": {
      const jobs = [...new Set([...s.jobs, ...event.jobs])]
      return [{ ...s, jobs, running: event.running && jobs.includes(event.running) ? event.running : s.running }, []]
    }
  }
}
