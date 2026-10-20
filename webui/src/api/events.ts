/** JSON events on /api/v2/ws (see docs/remi-webui/architecture.md). */
import type { FormTab, JobNote, JobOutput } from "./types"

type JobRef = {
  job_id: string
  /** Matches `job_seq` in binary preview frames. */
  seq?: number
  /** The job's form, when the server includes it. */
  tab?: FormTab
}

export type WsEventMap = {
  hello: { features: string[]; version?: string }
  "job.queued": JobRef & { position?: number }
  "job.started": JobRef & { steps?: number; job_count?: number }
  "job.progress": JobRef & { step: number; steps: number; eta?: number | null; job_no?: number; job_count?: number; text?: string }
  /** `status`: "completed", or "interrupted" / "cancelled" (an interrupted job keeps its results). */
  /** `warnings`: script steps that failed (the job went on without them). */
  "job.completed": JobRef & { outputs: JobOutput[]; infotext?: string | null; status?: string; warnings?: JobNote[] }
  "job.failed": JobRef & { error: string; status?: string }
  /** `running`: the job the worker runs now (also sent right after `hello`). */
  "queue.changed": { size?: number; paused?: boolean; running?: string | null }
  "options.changed": { keys?: string[] }
  /** A network folder was re-scanned (`kind`: lora, checkpoints, …). */
  "networks.changed": { kind?: string }
  /** Outputs were starred, removed or deleted from the gallery (`ids`: some of them). */
  "outputs.changed": { ids?: string[] }
  /** This client fell behind and lost `dropped` events; a fresh snapshot follows. */
  resync: { dropped?: number }
  pong: Record<string, never>
  /** A model load in progress; `stage` is free text. `done`, `error` and `failed` end it. */
  "model.loading": { stage: string; model?: string; progress?: number | null }
  /** A model load finished (successfully or not); ends `model.loading`. */
  "model.loaded": { model?: string; error?: string | null }
}

export type WsEventType = keyof WsEventMap
export type WsEvent = { [K in WsEventType]: { type: K } & WsEventMap[K] }[WsEventType]

type Fields = Record<string, unknown>

const isString = (v: unknown) => typeof v === "string"
const isNumber = (v: unknown) => typeof v === "number" && Number.isFinite(v)
const optional = (check: (v: unknown) => boolean) => (v: unknown) => v === undefined || v === null || check(v)
const isStringArray = (v: unknown) => Array.isArray(v) && v.every(isString)
const isOutputs = (v: unknown) =>
  Array.isArray(v) && v.every((o) => o !== null && typeof o === "object" && isString((o as Fields).id))

const JOB = { job_id: isString, seq: optional(isNumber) }

/** Required shape per event type: field → check. Unknown fields pass through. */
const SHAPES: { [K in WsEventType]: Record<string, (v: unknown) => boolean> } = {
  hello: { features: isStringArray },
  "job.queued": JOB,
  "job.started": { ...JOB, steps: optional(isNumber) },
  "job.progress": { ...JOB, step: isNumber, steps: isNumber },
  "job.completed": { ...JOB, outputs: optional(isOutputs) },
  "job.failed": { ...JOB, error: optional(isString) },
  "queue.changed": {},
  "options.changed": { keys: optional(isStringArray) },
  "networks.changed": { kind: optional(isString) },
  "outputs.changed": { ids: optional(isStringArray) },
  resync: { dropped: optional(isNumber) },
  pong: {},
  "model.loading": { stage: isString },
  "model.loaded": {},
}

/**
 * Validates a decoded text frame. Accepts both a flat `{type, ...fields}`
 * envelope and `{type, data: {...}}`; returns null for unknown types and
 * events missing a required field (they are logged in development).
 */
export function normalizeEvent(raw: unknown): WsEvent | null {
  if (!raw || typeof raw !== "object" || !("type" in raw) || typeof raw.type !== "string") return null
  const { data, ...rest } = raw as { type: string; data?: unknown }
  const event: Fields = { ...rest, ...(data && typeof data === "object" && !Array.isArray(data) ? data : {}) }
  const shape = Object.hasOwn(SHAPES, rest.type) ? SHAPES[rest.type as WsEventType] : undefined
  const bad = shape ? Object.entries(shape).find(([field, check]) => !check(event[field])) : undefined
  if (!shape || bad) {
    if (import.meta.env.DEV) console.warn(`[ws] ignored ${shape ? `malformed ${rest.type} (${bad?.[0]})` : `unknown event ${rest.type}`}`)
    return null
  }
  if (rest.type === "job.completed") event.outputs ??= []
  if (rest.type === "job.failed") event.error ??= "Failed"
  return event as WsEvent
}
