/**
 * Mock /api/v2 server (VITE_MOCK=1): answers requests from fixtures and
 * simulates jobs, emitting the same WebSocket events and binary preview
 * frames as the real server. Generated "images" are served by the dev
 * server's mock plugin (vite.config.ts) as SVG under /api/v2/files/mock-*.
 */
import { encodeFrame, FrameFormat } from "../frames"
import { buildIndex } from "@/ui/form-index"
import type { InputNode, ValidationError } from "@/ui/schema"
import { tabOwners } from "@/ui/tab"
import type { ActionRequest, Job, JobOutput, JobRequest } from "../types"
import { bootstrap, extrasSchema, models, serverCatalogZh, tabSchema, toolboxPage } from "./fixtures"
import { broadcast, broadcastBinary } from "./socket"

type Handler = (match: RegExpExecArray, body: unknown, url: URL) => unknown | Promise<unknown>

/** Thrown by a handler to answer with an error status and JSON body. */
class MockHttpError extends Error {
  readonly status: number
  readonly body: unknown
  constructor(status: number, body: unknown) {
    super(`HTTP ${status}`)
    this.status = status
    this.body = body
  }
}

const jobs: Job[] = []
let seq = 0
let uploadSeq = 0
let running: { id: string; cancel: () => void } | null = null
const queue: { job: Job; request: JobRequest }[] = []
const requests = new Map<string, JobRequest>()

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

function json(data: unknown, status = 200): Response {
  return new Response(data === undefined ? null : JSON.stringify(data), {
    status: data === undefined ? 204 : status,
    headers: { "Content-Type": "application/json" },
  })
}

// ---------------------------------------------------------------- actions

type Values = Record<string, unknown>

const PRESETS: Record<string, [sampler: string, steps: number, cfg: number]> = {
  balanced: ["Euler a", 28, 5],
  detail: ["DPM++ 2M", 40, 4],
  speed: ["Euler a", 16, 4.5],
}

/** The Hires. fix output line (`remi.core.actions.hires_resolution_html`). */
function hiresLine(core: Values): string {
  if (!core.enable_hr) return ""
  const width = Number(core.width ?? 1024)
  const height = Number(core.height ?? 1024)
  const scale = Number(core.hr_scale ?? 2)
  const snap = (n: number) => Math.round(n / 8) * 8
  return `from <span class='resolution'>${width}x${height}</span> to <span class='resolution'>${snap(width * scale)}x${snap(height * scale)}</span>`
}

/**
 * A stand-in for `remi.core.conflicts.conflict_patch`, over the whole form
 * (`values` = core, `scripts` = the other owners). Keys are relative to core.
 * Rules: the Anima locks are one-of-two; Hires. fix and the refiner exclude
 * each other (the last one switched on wins); LCM pins the SGM Uniform schedule.
 */
function conflictPatch(core: Values, scripts: Record<string, Values>, selected: string | undefined) {
  const patch: Record<string, unknown> = {}
  const anima = scripts.anima ?? {}
  if (anima.lock_steps && anima.lock_cfg) {
    const keep = selected === "anima.lock_cfg" ? "lock_cfg" : "lock_steps"
    patch[`anima.${keep === "lock_cfg" ? "lock_steps" : "lock_cfg"}`] = { value: false }
  }
  const refiner = scripts.refiner ?? {}
  if (core.enable_hr && refiner.enable) {
    if (selected === "refiner.enable") patch.enable_hr = { value: false }
    else patch["refiner.enable"] = { value: false }
  }
  const lcm = scripts.sampler?.sampler_name === "LCM"
  patch["sampler.scheduler"] = lcm ? { value: "SGM Uniform", enabled: false } : { enabled: true }
  if (lcm && selected === "sampler.sampler_name") {
    patch.__toast__ = { level: "warning", message: "Compatibility: LCM uses the SGM Uniform schedule" }
  }
  return patch
}

function actionPatch(owner: string, action: string, body: ActionRequest & { selected?: string }) {
  const values = body.values ?? {}
  const scripts = body.scripts ?? {}
  switch (`${owner}/${action}`) {
    case "core/swap_dimensions":
      return { width: { value: values.height }, height: { value: values.width } }
    case "core/hr_resolution":
      return { __outputs__: { hr_final_resolution: hiresLine(values) } }
    case "core/__conflicts__":
      return conflictPatch(values, scripts, body.selected)
    case "anima/apply_preset": {
      const [sampler, steps, cfg] = PRESETS[String(values.preset)] ?? PRESETS.balanced ?? ["Euler a", 28, 5]
      return {
        enable: { value: true },
        "sampler.sampler_name": { value: sampler },
        "sampler.steps": { value: steps },
        "core.cfg_scale": { value: cfg },
        __toast__: { level: "success", message: `Preset “${String(values.preset)}” applied` },
        __outputs__: { status: `Sampler ${sampler}, ${steps} steps, CFG ${cfg}` },
      }
    }
    case "anima/recommended":
      return { shift: { value: 3 }, rescale: { value: 0 }, __toast__: { level: "info", message: "Recommended values restored" } }
    case "controlnet/model_changed":
      return values.model === "None"
        ? { weight: { enabled: false, info: null } }
        : { weight: { enabled: true, info: `Recommended weight for ${String(values.model)}: 0.8` } }
    case "batch-tools/analyze": {
      const refs = Array.isArray(values.references) ? values.references : []
      return {
        __outputs__: {
          analysis: refs.length > 0 ? refs : ["mock-analysis-1-200", "mock-analysis-2-320"],
          report: { images: refs.length, average_brightness: 0.42 },
          summary: "<p><strong>Done.</strong> No issues found.</p>",
        },
      }
    }
    case "page:toolbox/run": {
      const text = String(values.text ?? "")
      const result =
        values.transform === "lower" ? text.toLowerCase() : values.transform === "reverse" ? [...text].reverse().join("") : text.toUpperCase()
      return { __outputs__: { result } }
    }
    default:
      return { __toast__: { level: "warning", message: `Mock: no handler for ${owner}/${action}` } }
  }
}

// ---------------------------------------------------------------- validation

/**
 * A small version of the server's request validation (schema-format.md §3):
 * numbers must be finite and in range, choices must be listed, plus two
 * demo rules (img2img needs a source image; "#invalid" in the prompt).
 */
export function validateJob(request: JobRequest): ValidationError[] {
  const schema = request.tab === "extras" ? extrasSchema : tabSchema(request.tab)
  const index = buildIndex(tabOwners(schema))
  const errors: ValidationError[] = []
  const owners: [string, Values][] = [["core", request.core], ...Object.entries(request.scripts ?? {})]
  for (const [owner, values] of owners) {
    for (const [name, value] of Object.entries(values ?? {})) {
      const node = index.inputs.get(owner)?.get(name)
      const message = node ? invalid(node, value) : null
      if (message) errors.push({ owner, param: name, message })
    }
  }
  if (request.tab === "img2img" && (request.core.mode ?? "img2img") === "img2img" && !request.core.init_image) {
    errors.push({ owner: "core", param: "init_image", message: "an image is required" })
  }
  if (String(request.core.prompt ?? "").includes("#invalid")) {
    errors.push({ owner: "core", param: "prompt", message: "the prompt contains #invalid (mock validation demo)" })
  }
  return errors
}

function invalid(node: InputNode, value: unknown): string | null {
  switch (node.type) {
    case "slider":
    case "number": {
      if (typeof value !== "number" || !Number.isFinite(value)) return `expected a number, got ${JSON.stringify(value)}`
      if (node.type === "number" && node.minimum != null && value < node.minimum) return `must be at least ${node.minimum}`
      if (node.type === "number" && node.maximum != null && value > node.maximum) return `must be at most ${node.maximum}`
      return null
    }
    case "seed":
      return Number.isInteger(value) ? null : `expected an integer, got ${JSON.stringify(value)}`
    case "choice":
      return node.allow_custom || node.choices.some((c) => c.value === value) ? null : `${JSON.stringify(value)} is not one of the choices`
    default:
      return null
  }
}

// ---------------------------------------------------------------- jobs

async function previewFrame(jobSeq: number, step: number, steps: number, hue: number): Promise<ArrayBuffer | null> {
  if (typeof document === "undefined") return null
  const canvas = document.createElement("canvas")
  canvas.width = 384
  canvas.height = 384
  const ctx = canvas.getContext("2d")
  if (!ctx) return null
  const t = step / steps
  const gradient = ctx.createLinearGradient(0, 0, 384, 384)
  gradient.addColorStop(0, `hsl(${hue} 70% ${20 + 40 * t}%)`)
  gradient.addColorStop(1, `hsl(${(hue + 80) % 360} 70% ${15 + 30 * t}%)`)
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, 384, 384)
  // Noise that fades as denoising progresses.
  for (let i = 0; i < 1400 * (1 - t); i++) {
    ctx.fillStyle = `hsla(${(hue + i) % 360} 80% 60% / ${0.6 * (1 - t)})`
    ctx.fillRect(Math.random() * 384, Math.random() * 384, 3, 3)
  }
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/webp", 0.7))
  if (!blob) return null
  return encodeFrame(blob.type === "image/webp" ? FrameFormat.Webp : FrameFormat.Jpeg, jobSeq, step, new Uint8Array(await blob.arrayBuffer()))
}

function infotext(request: JobRequest, seed: number): string {
  const c = request.core
  const sampler = request.scripts.sampler ?? {}
  const size = `${String(c.width ?? 1024)}x${String(c.height ?? 1024)}`
  return `${String(c.prompt ?? "")}\nNegative prompt: ${String(c.negative_prompt ?? "")}\nSteps: ${String(sampler.steps ?? 20)}, Sampler: ${String(sampler.sampler_name ?? "Euler a")}, Schedule type: ${String(sampler.scheduler ?? "Automatic")}, CFG scale: ${String(c.cfg_scale ?? 5)}, Seed: ${seed}, Size: ${size}, Model: mock`
}

async function runJob(job: Job, request: JobRequest) {
  let cancelled = false
  running = { id: job.id, cancel: () => (cancelled = true) }
  job.status = "running"
  job.started_at = new Date().toISOString()
  const steps = Math.min(Number(request.scripts.sampler?.steps ?? 20), 60)
  const count = Math.max(1, Number(request.core.batch_size ?? 1))
  const hue = Math.floor(Math.random() * 360)
  broadcast({ type: "job.started", job_id: job.id, seq: job.seq, steps, job_count: 1 })
  broadcast({ type: "queue.changed" })

  for (let step = 1; step <= steps && !cancelled; step++) {
    await delay(90)
    const eta = ((steps - step) * 90) / 1000
    job.progress = { step, steps, eta }
    broadcast({ type: "job.progress", job_id: job.id, seq: job.seq, step, steps, eta, job_no: 0, job_count: 1 })
    if (step % 3 === 0 || step === steps) {
      const frame = await previewFrame(job.seq ?? 0, step, steps, hue)
      if (frame) broadcastBinary(frame)
    }
  }

  running = null
  job.finished_at = new Date().toISOString()
  if (cancelled) {
    job.status = "interrupted"
    broadcast({ type: "job.failed", job_id: job.id, seq: job.seq, error: "Interrupted" })
  } else {
    const seedValue = Number(request.scripts.seed?.seed ?? -1)
    const seed = seedValue === -1 ? Math.floor(Math.random() * 2 ** 32) : seedValue
    const outputs: JobOutput[] = Array.from({ length: count }, (_, i) => ({
      id: `mock-${job.seq}-${i}-${(hue + i * 40) % 360}`,
      kind: "image",
      width: Number(request.core.width ?? 1024),
      height: Number(request.core.height ?? 1024),
    }))
    job.status = "completed"
    job.outputs = outputs
    broadcast({ type: "job.completed", job_id: job.id, seq: job.seq, outputs, infotext: infotext(request, seed) })
  }
  broadcast({ type: "queue.changed" })
  const next = queue.shift()
  if (next) void runJob(next.job, next.request)
}

function enqueue(request: JobRequest) {
  const job: Job = {
    id: `job-${++seq}`,
    seq,
    tab: request.tab,
    status: "queued",
    created_at: new Date().toISOString(),
    outputs: [],
    summary: String(request.core.prompt ?? "").slice(0, 120),
  }
  jobs.unshift(job)
  requests.set(job.id, request)
  broadcast({ type: "job.queued", job_id: job.id, seq: job.seq, tab: job.tab, position: queue.length })
  if (running) queue.push({ job, request })
  else void runJob(job, request)
  return { job_id: job.id }
}

// ---------------------------------------------------------------- routes

const routes: [method: string, pattern: RegExp, handler: Handler][] = [
  ["GET", /^\/app$/, () => bootstrap],
  ["GET", /^\/schema\/(txt2img|img2img|extras)$/, (m) => (m[1] === "extras" ? extrasSchema : tabSchema(m[1] as "txt2img" | "img2img"))],
  ["GET", /^\/pages\/toolbox$/, () => toolboxPage],
  ["GET", /^\/models$/, () => models],
  ["GET", /^\/i18n\/([\w-]+)$/, (m) => (m[1] === "zh_CN" ? serverCatalogZh : {})],
  ["GET", /^\/options$/, () => ({ version: 1, categories: [], quicksettings: ["sd_model_checkpoint"], values: bootstrap.options })],
  [
    "PATCH",
    /^\/options$/,
    async (_m, body) => {
      await delay(400)
      Object.assign(bootstrap.options, body)
      if (typeof (body as Record<string, unknown>).sd_model_checkpoint === "string") {
        bootstrap.models.checkpoint = (body as Record<string, string>).sd_model_checkpoint
      }
      broadcast({ type: "options.changed", keys: Object.keys(body as object) })
      return { version: 1, categories: [], quicksettings: [], values: bootstrap.options }
    },
  ],
  ["POST", /^\/actions\/([^/]+)\/([^/]+)$/, async (m, body) => {
    await delay(150)
    return actionPatch(decodeURIComponent(m[1] ?? ""), decodeURIComponent(m[2] ?? ""), body as ActionRequest)
  }],
  [
    "POST",
    /^\/jobs$/,
    (_m, body) => {
      const request = body as JobRequest
      const errors = validateJob(request)
      if (errors.length > 0) throw new MockHttpError(422, { errors })
      return enqueue(request)
    },
  ],
  [
    "GET",
    /^\/jobs$/,
    (_m, _body, url) => {
      // Cursor = offset into the newest-first list.
      const offset = Number(url.searchParams.get("cursor") ?? 0) || 0
      const limit = Math.min(Number(url.searchParams.get("limit") ?? 50) || 50, 200)
      const items = jobs.slice(offset, offset + limit)
      return { items, next_cursor: offset + limit < jobs.length ? String(offset + limit) : null, total: jobs.length }
    },
  ],
  [
    "POST",
    /^\/jobs\/([^/]+)\/retry$/,
    (m) => {
      const request = requests.get(m[1] ?? "")
      if (!request) throw new MockHttpError(404, { detail: "Mock: unknown job" })
      return enqueue(request)
    },
  ],
  [
    "POST",
    /^\/jobs\/([^/]+)\/(interrupt|skip)$/,
    (m) => {
      if (running && running.id === m[1]) running.cancel()
      return undefined
    },
  ],
  [
    "DELETE",
    /^\/jobs\/([^/]+)$/,
    (m) => {
      const index = jobs.findIndex((j) => j.id === m[1])
      if (index >= 0 && jobs[index]?.status !== "running") jobs.splice(index, 1)
      const queued = queue.findIndex((q) => q.job.id === m[1])
      if (queued >= 0) queue.splice(queued, 1)
      broadcast({ type: "queue.changed" })
      return undefined
    },
  ],
  [
    "POST",
    /^\/uploads$/,
    async (_m, body) => {
      await delay(250)
      const file = body instanceof FormData ? body.get("file") : null
      return { id: `upload-${++uploadSeq}`, name: file instanceof File ? file.name : "upload", width: null, height: null }
    },
  ],
]

export async function handleMockRequest(request: Request): Promise<Response> {
  const url = new URL(request.url, location.origin)
  const path = url.pathname.replace(/^\/api\/v2/, "")
  for (const [method, pattern, handler] of routes) {
    if (method !== request.method) continue
    const match = pattern.exec(path)
    if (!match) continue
    const type = request.headers.get("Content-Type") ?? ""
    const body = type.includes("application/json") ? await request.json() : type.includes("multipart") ? await request.formData() : undefined
    try {
      return json(await handler(match, body, url))
    } catch (error) {
      if (error instanceof MockHttpError) return json(error.body, error.status)
      return json({ detail: error instanceof Error ? error.message : String(error) }, 500)
    }
  }
  return json({ detail: `Mock: ${request.method} ${path} not found` }, 404)
}
