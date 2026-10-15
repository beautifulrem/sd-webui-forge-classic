/** Complete API models for tests (the generated types require every field the server always sends). */
import type { Bootstrap, Job, JobCreated, JobDetail, JobOutput } from "@/api/types"

/** `POST /jobs` answer for one job. */
export const created = (id: string, seq = 1): JobCreated => ({ job_id: id, job_ids: [id], seq })

export const output = (id: string, extra: Partial<JobOutput> = {}): JobOutput => ({ id, kind: "image", ...extra })

export const jobOf = (id: string, status: Job["status"], outputs: JobOutput[] = []): Job => ({
  id,
  seq: 1,
  tab: "txt2img",
  status,
  created_at: "2026-10-15T00:00:00Z",
  outputs,
  priority: 0,
  forever: false,
  starred: false,
  replayable: true,
  notes: [],
  warnings: [],
})

/** `GET /jobs/{id}`: a job with its request. */
export const jobDetail = (id: string, status: Job["status"], outputs: JobOutput[] = []): JobDetail => ({
  ...jobOf(id, status, outputs),
  core: {},
  scripts: {},
  layout_hash: "",
  schema_version: 1,
})

/** A bootstrap with only `options` set. */
export const bootstrapWith = (options: Record<string, unknown>): Bootstrap => ({
  version: "t",
  features: [],
  tabs: [],
  pages: [],
  locales: [],
  options,
  models: { text_encoders: [], modules: [] },
  not_ported: [],
  auth: { required: false },
  queue: { size: 0, paused: false },
  user_assets: [],
})
