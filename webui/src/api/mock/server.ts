/**
 * Mock /api/v2 server (VITE_MOCK=1): answers requests from fixtures and
 * simulates jobs, emitting the same WebSocket events and binary preview
 * frames as the real server. Generated "images" are served by the dev
 * server's mock plugin (vite.config.ts) as SVG under /api/v2/files/mock-*.
 */
import { materializeStyles } from "@/features/prompt-editor/styles/apply-styles"
import { encodeFrame, FrameFormat } from "../frames"
import { buildIndex } from "@/ui/form-index"
import type { InputNode, ValidationError } from "@/ui/schema"
import { tabOwners } from "@/ui/tab"
import type { ActionRequest, FormTab, InfotextParsed, Job, JobOutput, JobRequest, PromptStyle, TokenCountRequest } from "../types"
import { bootstrap, extrasSchema, models, serverCatalogZh, styles, tabSchema, toolboxPage } from "./fixtures"
import { broadcast, broadcastBinary } from "./socket"
import { adminRoutes } from "./admin"
import { networkRoutes } from "./networks"
import { libraryRoutes } from "./library"

type Handler = (match: RegExpExecArray, body: unknown, url: URL) => unknown | Promise<unknown>

/** Thrown by a handler to answer with an error status and JSON body. */
export class MockHttpError extends Error {
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

/** UI preset → [size, cfg, sampler, steps] (a stand-in for `forge_preset_patch`). */
const FORGE_PRESETS: Record<string, [number, number, string, number]> = {
  sd: [512, 7, "Euler a", 20],
  xl: [1024, 5, "Euler a", 28],
  flux: [1024, 1, "Euler", 20],
  anima: [1024, 4, "ER SDE", 30],
}

/** Emits the `model.loading` stages a real model switch produces. */
function simulateModelLoad(model: string) {
  const stages = ["Unloading previous model", "Loading state dict", "Moving model to device"]
  stages.forEach((stage, index) => {
    setTimeout(() => broadcast({ type: "model.loading", stage, model, progress: (index + 1) / (stages.length + 1) }), 150 + index * 450)
  })
  setTimeout(() => broadcast({ type: "model.loaded", model }), 150 + stages.length * 450)
}

/** The Hires. fix output line (`remi.core.actions.hires_resolution`): a translatable message. */
function hiresLine(core: Values): { text: string; params: Record<string, string> } | "" {
  if (!core.enable_hr) return ""
  const width = Number(core.width ?? 1024)
  const height = Number(core.height ?? 1024)
  const scale = Number(core.hr_scale ?? 2)
  const snap = (n: number) => Math.round(n / 8) * 8
  return { text: "from {{from}} to {{to}}", params: { from: `${width}×${height}`, to: `${snap(width * scale)}×${snap(height * scale)}` } }
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
    case "core/__forge_preset__": {
      const preset = String((body as { preset?: unknown }).preset ?? "sd")
      const [size, cfg, sampler, steps] = FORGE_PRESETS[preset] ?? FORGE_PRESETS.sd!
      bootstrap.options.forge_preset = preset
      simulateModelLoad(`${preset} preset`)
      broadcast({ type: "options.changed", keys: ["forge_preset"] })
      return {
        width: { value: size },
        height: { value: size },
        cfg_scale: { value: cfg },
        "sampler.sampler_name": { value: sampler },
        "sampler.steps": { value: steps },
        __settings__: { forge_preset: preset },
      }
    }
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
  if (request.tab === "img2img" && !request.core.init_img) {
    errors.push({ owner: "core", param: "init_img", message: "an image is required" })
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
  // Like the WebUI: the Hires. fix values only when it ran (txt2img).
  const hires = request.tab === "txt2img" && c.enable_hr ? `, Denoising strength: ${String(c.denoising_strength ?? 0.7)}, Hires upscale: ${String(c.hr_scale ?? 2)}, Hires upscaler: ${String(c.hr_upscaler ?? "Latent")}` : ""
  // The refiner script's values when it is on with a checkpoint (generation tabs).
  const refinerValues = request.scripts.refiner ?? {}
  const refiner = request.tab !== "extras" && refinerValues.enable && refinerValues.checkpoint ? `, Refiner: ${String(refinerValues.checkpoint)}, Refiner switch at: ${String(refinerValues.switch_at ?? 0.8)}` : ""
  return `${String(c.prompt ?? "")}\nNegative prompt: ${String(c.negative_prompt ?? "")}\nSteps: ${String(sampler.steps ?? 20)}, Sampler: ${String(sampler.sampler_name ?? "Euler a")}, Schedule type: ${String(sampler.scheduler ?? "Automatic")}, CFG scale: ${String(c.cfg_scale ?? 5)}, Seed: ${seed}, Size: ${size}${hires}${refiner}, Model: mock`
}

/** Sizes of uploads (where the browser could decode them) and of outputs, by file id: Extras scales them. */
const imageSizes = new Map<string, { width: number; height: number }>()

async function imageSize(file: Blob): Promise<{ width: number; height: number } | null> {
  if (typeof createImageBitmap !== "function") return null
  try {
    const bitmap = await createImageBitmap(file)
    const size = { width: bitmap.width, height: bitmap.height }
    bitmap.close()
    return size
  } catch {
    return null
  }
}

/** An output's size: Extras scales its source, Hires. fix the generated size; else the requested size. */
function outputSize(request: JobRequest): { width: number; height: number } {
  const c = request.core
  if (request.tab === "extras") {
    const source = imageSizes.get(String(c.image ?? "")) ?? { width: 512, height: 512 }
    const scale = Number(c.upscaling_resize ?? 2)
    return { width: Math.round(source.width * scale), height: Math.round(source.height * scale) }
  }
  const scale = request.tab === "txt2img" && c.enable_hr ? Number(c.hr_scale ?? 2) : 1
  return { width: Math.round(Number(c.width ?? 1024) * scale), height: Math.round(Number(c.height ?? 1024) * scale) }
}

/** Extras' infotext: the operations' parameters (as `run_extras_named` returns them). */
function extrasInfotext(request: JobRequest): string {
  return `Postprocess upscale by: ${String(request.core.upscaling_resize ?? 2)}, Postprocess upscaler: ${String(request.core.upscaler_1 ?? "None")}`
}

/** Like `GET /queue`; also sent to every socket right after its hello. */
let paused = false

export function queueStatus() {
  return { size: queue.length, paused, running: running?.id ?? null }
}

async function runJob(job: Job, request: JobRequest) {
  let cancelled = false
  running = { id: job.id, cancel: () => (cancelled = true) }
  job.status = "running"
  job.started_at = new Date().toISOString()
  const steps = request.tab === "extras" ? 5 : Math.min(Number(request.scripts.sampler?.steps ?? 20), 60)
  const count = Math.max(1, Number(request.core.batch_size ?? 1))
  const hue = Math.floor(Math.random() * 360)
  broadcast({ type: "job.started", job_id: job.id, seq: job.seq, steps, job_count: 1 })
  broadcast({ type: "queue.changed", ...queueStatus() })

  // oxlint-disable-next-line eslint/no-unmodified-loop-condition -- the interrupt handler sets `cancelled` while this awaits
  for (let step = 1; step <= steps && !cancelled; step++) {
    await delay(90)
    const eta = ((steps - step) * 90) / 1000
    job.progress = { step, steps, eta, job_no: 0, job_count: 1 }
    broadcast({ type: "job.progress", job_id: job.id, seq: job.seq, step, steps, eta, job_no: 0, job_count: 1 })
    if (step % 3 === 0 || step === steps) {
      const frame = await previewFrame(job.seq ?? 0, step, steps, hue)
      if (frame) broadcastBinary(frame)
    }
  }

  running = null
  job.finished_at = new Date().toISOString()
  if (cancelled) {
    // Like the server: an interrupted job ends as completed with status "interrupted".
    job.status = "interrupted"
    broadcast({ type: "job.completed", job_id: job.id, seq: job.seq, tab: job.tab, status: "interrupted", outputs: [] })
  } else {
    const seedValue = Number(request.scripts.seed?.seed ?? -1)
    const seed = seedValue === -1 ? Math.floor(Math.random() * 2 ** 32) : seedValue
    const outputs: JobOutput[] = Array.from({ length: count }, (_, i) => ({
      id: `mock-${job.seq}-${i}-${(hue + i * 40) % 360}`,
      kind: "image",
      ...outputSize(request),
    }))
    const text = request.tab === "extras" ? extrasInfotext(request) : infotext(request, seed)
    for (const output of outputs) imageSizes.set(output.id, { width: output.width ?? 0, height: output.height ?? 0 })
    for (const output of outputs) output.infotext = text
    if (request.tab !== "extras") lastInfotext = text // like params.txt: generations only
    job.status = "completed"
    job.outputs = outputs
    broadcast({ type: "job.completed", job_id: job.id, seq: job.seq, tab: job.tab, status: "completed", outputs, infotext: text })
  }
  broadcast({ type: "queue.changed", ...queueStatus() })
  startNext()
}

/** Runs the next queued job unless one runs or the queue is paused. */
function startNext() {
  if (running || paused) return
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
    priority: request.priority ?? 0,
    forever: request.forever ?? false,
    summary: String(request.core.prompt ?? "").slice(0, 120),
    starred: false,
    replayable: true,
    notes: [],
    warnings: [],
  }
  jobs.unshift(job)
  requests.set(job.id, request)
  broadcast({ type: "job.queued", job_id: job.id, seq: job.seq, tab: job.tab, position: queue.length })
  queue.push({ job, request })
  startNext()
  return job
}

function enqueueAll(request: JobRequest) {
  const created = Array.from({ length: Math.max(1, Math.min(100, Number(request.repeat ?? 1))) }, () => enqueue(request))
  return { job_id: created[0]!.id, job_ids: created.map((job) => job.id), seq: created[0]!.seq }
}

// ---------------------------------------------------------------- prompt tools

/** A rough CLIP-like count: words and punctuation, extra networks and BREAK excluded. */
function countTokens({ text }: TokenCountRequest) {
  const clean = text.replace(/<[^<>\n]*>/g, " ").replace(/\bBREAK\b/g, " ")
  const tokens = clean.match(/[\p{L}\p{N}]+|[^\s\p{L}\p{N}]/gu)?.length ?? 0
  return { tokens, max_length: Math.max(75, Math.ceil(tokens / 75) * 75) }
}

let lastInfotext = ""

const PARAM_LINE = /\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)/g

/** A stand-in for `remi.core.infotext.parse_infotext` over the mock schema's names. */
function parseInfotextMock(text: string, tab: FormTab | null | undefined): InfotextParsed {
  const source = text.trim() || lastInfotext
  const lines = source.split("\n")
  const params: Record<string, string> = {}
  const last = lines.at(-1) ?? ""
  const pairs = [...last.matchAll(PARAM_LINE)]
  if (pairs.length >= 3) {
    lines.pop()
    for (const m of pairs) params[m[1]!.trim()] = m[2]!.trim().replace(/^"(.*)"$/, "$1")
  }
  const negativeAt = lines.findIndex((line) => line.startsWith("Negative prompt:"))
  const prompt = (negativeAt === -1 ? lines : lines.slice(0, negativeAt)).join("\n").trim()
  const negative = negativeAt === -1 ? "" : lines.slice(negativeAt).join("\n").replace(/^Negative prompt:\s*/, "").trim()
  params.Prompt = prompt
  if (negativeAt !== -1) params["Negative prompt"] = negative

  const detected: FormTab = tab ?? ("Denoising strength" in params && !("Hires upscale" in params) ? "img2img" : "txt2img")
  const core: Record<string, unknown> = {}
  const scripts: Record<string, Record<string, unknown>> = {}
  const problems: InfotextParsed["problems"] = []
  const num = (key: string) => (key in params ? Number(params[key]) : undefined)
  if (prompt) core.prompt = prompt
  if (negativeAt !== -1) core.negative_prompt = negative
  if (num("CFG scale") !== undefined) core.cfg_scale = num("CFG scale")
  const size = /^(\d+)x(\d+)$/.exec(params.Size ?? "")
  if (size) Object.assign(core, { width: Number(size[1]), height: Number(size[2]) })
  if (num("Denoising strength") !== undefined) core.denoising_strength = num("Denoising strength")
  if (detected === "txt2img" && num("Hires upscale") !== undefined) Object.assign(core, { enable_hr: true, hr_scale: num("Hires upscale") })
  const sampler: Record<string, unknown> = {}
  if (num("Steps") !== undefined) sampler.steps = num("Steps")
  if (params.Sampler) {
    if (models.samplers.some((s) => s.name === params.Sampler)) sampler.sampler_name = params.Sampler
    else problems.push({ owner: "sampler", param: "sampler_name", message: `unknown sampler ${params.Sampler}` })
  }
  if (params["Schedule type"]) sampler.scheduler = params["Schedule type"]
  if (Object.keys(sampler).length > 0) scripts.sampler = sampler
  if (num("Seed") !== undefined) scripts.seed = { seed: num("Seed") }
  return { tab: detected, core, scripts, params, problems }
}

// ---------------------------------------------------------------- routes

const routes: [method: string, pattern: RegExp, handler: Handler][] = [
  ...adminRoutes, // settings & extensions pages (admin.ts)
  ...networkRoutes,
  ...libraryRoutes, // gallery, PNG info & queue tools (library.ts)
  ["GET", /^\/app$/, () => bootstrap],
  ["GET", /^\/schema\/(txt2img|img2img|extras)$/, (m) => (m[1] === "extras" ? extrasSchema : tabSchema(m[1] as "txt2img" | "img2img"))],
  ["GET", /^\/pages\/toolbox$/, () => toolboxPage],
  ["GET", /^\/models$/, () => models],
  ["GET", /^\/i18n\/([\w-]+)$/, (m) => (m[1] === "zh_CN" ? serverCatalogZh : {})],
  [
    "POST",
    /^\/prompt\/tokens$/,
    async (_m, body) => {
      await delay(60)
      return countTokens(body as TokenCountRequest)
    },
  ],
  ["POST", /^\/infotext\/parse$/, (_m, body) => parseInfotextMock(String((body as { text?: unknown }).text ?? ""), (body as { tab?: FormTab }).tab)],
  ["GET", /^\/styles$/, () => styles],
  [
    "POST",
    /^\/styles\/apply$/,
    (_m, body) => {
      const b = body as { prompt?: string; negative_prompt?: string; styles?: string[] }
      const applied = materializeStyles({ prompt: b.prompt ?? "", negative: b.negative_prompt ?? "" }, b.styles ?? [], styles)
      return { prompt: applied.prompt, negative_prompt: applied.negative }
    },
  ],
  [
    "POST",
    /^\/infotext\/image$/,
    (_m, body) => {
      // Mock outputs carry their infotext; uploads have none.
      const id = String((body as { file_id?: unknown }).file_id ?? "")
      const text = jobs.flatMap((job) => job.outputs).find((output) => output.id === id)?.infotext
      return text ? { infotext: text, items: {}, parsed: parseInfotextMock(text, (body as { tab?: FormTab }).tab) } : { infotext: null, items: {}, parsed: null }
    },
  ],
  [
    "PUT",
    /^\/styles\/([^/]+)$/,
    (m, body) => {
      const name = decodeURIComponent(m[1] ?? "")
      if (!name.trim()) throw new MockHttpError(422, { detail: "A style needs a name" })
      const style: PromptStyle = { name, prompt: String((body as PromptStyle).prompt ?? ""), negative_prompt: String((body as PromptStyle).negative_prompt ?? "") }
      const index = styles.findIndex((s) => s.name === name)
      if (index === -1) styles.push(style)
      else styles[index] = style
      return style
    },
  ],
  [
    "DELETE",
    /^\/styles\/([^/]+)$/,
    (m) => {
      const index = styles.findIndex((s) => s.name === decodeURIComponent(m[1] ?? ""))
      if (index === -1) throw new MockHttpError(404, { detail: "Mock: unknown style" })
      styles.splice(index, 1)
      return undefined
    },
  ],
  ["GET", /^\/options$/, () => ({ version: 1, categories: [], quicksettings: ["sd_model_checkpoint"], values: bootstrap.options })],
  [
    "PATCH",
    /^\/options$/,
    async (_m, body) => {
      await delay(400)
      Object.assign(bootstrap.options, body)
      const changed = body as Record<string, unknown>
      if (typeof changed.sd_model_checkpoint === "string") bootstrap.models.checkpoint = changed.sd_model_checkpoint
      if (["sd_model_checkpoint", "forge_additional_modules", "forge_unet_storage_dtype"].some((key) => key in changed)) {
        simulateModelLoad(String(bootstrap.options.sd_model_checkpoint))
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
      return enqueueAll(request)
    },
  ],
  [
    "GET",
    /^\/jobs$/,
    (_m, _body, url) => {
      // Cursor = offset into the newest-first list.
      const offset = Number(url.searchParams.get("cursor") ?? 0) || 0
      const limit = Math.min(Number(url.searchParams.get("limit") ?? 50) || 50, 200)
      const statuses = url.searchParams.get("status")?.split(",")
      const terms = (url.searchParams.get("q") ?? "").toLowerCase().split(/\s+/).filter(Boolean)
      const starred = url.searchParams.get("starred")
      const tab = url.searchParams.get("tab")
      const matching = jobs.filter(
        (job) =>
          (!statuses || statuses.includes(job.status)) &&
          (!tab || job.tab === tab) &&
          (starred === null || job.starred === (starred === "true")) &&
          terms.every((term) => `${job.summary ?? ""} ${job.infotext ?? ""}`.toLowerCase().includes(term)),
      )
      const items = matching.slice(offset, offset + limit)
      return { items, next_cursor: offset + limit < matching.length ? String(offset + limit) : null, total: matching.length }
    },
  ],
  [
    "GET",
    /^\/jobs\/([^/]+)$/,
    (m) => {
      const job = jobs.find((j) => j.id === m[1])
      if (!job) throw new MockHttpError(404, { detail: "Mock: unknown job" })
      const request = requests.get(job.id)
      return { ...job, core: request?.core ?? {}, scripts: request?.scripts ?? {}, layout_hash: "mock", schema_version: 1 }
    },
  ],
  ["GET", /^\/queue\/items$/, () => queue.map((q) => q.job)],
  ["GET", /^\/queue$/, () => queueStatus()],
  [
    "POST",
    /^\/queue\/(pause|resume)$/,
    (m) => {
      paused = m[1] === "pause"
      broadcast({ type: "queue.changed", ...queueStatus() })
      startNext()
      return queueStatus()
    },
  ],
  [
    "POST",
    /^\/queue\/reorder$/,
    (_m, body) => {
      const order = (body as { order?: string[] }).order ?? []
      const rank = (id: string) => (order.includes(id) ? order.indexOf(id) : -1)
      const listed = queue.filter((q) => rank(q.job.id) !== -1).sort((a, b) => rank(a.job.id) - rank(b.job.id))
      let next = 0
      for (const [index, entry] of queue.entries()) if (rank(entry.job.id) !== -1) queue[index] = listed[next++]!
      queue.sort((a, b) => b.job.priority - a.job.priority)
      broadcast({ type: "queue.changed", ...queueStatus() })
      return queueStatus()
    },
  ],
  [
    "POST",
    /^\/queue\/move$/,
    (_m, body) => {
      const { id, after, before, priority } = body as { id: string; after?: string | null; before?: string | null; priority?: number | null }
      const from = queue.findIndex((q) => q.job.id === id)
      if (from === -1) throw new MockHttpError(409, { detail: "Mock: the job is not queued any more" })
      const [entry] = queue.splice(from, 1)
      if (typeof priority === "number") entry!.job.priority = priority
      const anchor = after ? queue.findIndex((q) => q.job.id === after) : -1
      const below = before ? queue.findIndex((q) => q.job.id === before) : -1
      const at = anchor !== -1 ? anchor + 1 : below !== -1 ? below : from
      queue.splice(at, 0, entry!)
      queue.sort((a, b) => b.job.priority - a.job.priority)
      broadcast({ type: "queue.changed", ...queueStatus() })
      return queueStatus()
    },
  ],
  [
    "PATCH",
    /^\/jobs\/([^/]+)$/,
    (m, body) => {
      const job = jobs.find((j) => j.id === m[1])
      if (!job) throw new MockHttpError(404, { detail: "Mock: unknown job" })
      const patch = body as { priority?: number; forever?: boolean }
      if (typeof patch.priority === "number") job.priority = patch.priority
      if (typeof patch.forever === "boolean") job.forever = patch.forever
      queue.sort((a, b) => b.job.priority - a.job.priority)
      broadcast({ type: "queue.changed", ...queueStatus() })
      return job
    },
  ],
  [
    "POST",
    /^\/jobs\/([^/]+)\/duplicate$/,
    (m) => {
      const request = requests.get(m[1] ?? "")
      if (!request) throw new MockHttpError(404, { detail: "Mock: unknown job" })
      return enqueueAll({ ...request, repeat: 1 })
    },
  ],
  [
    "PUT",
    /^\/jobs\/([^/]+)\/params$/,
    (m, body) => {
      const job = jobs.find((j) => j.id === m[1])
      if (!job) throw new MockHttpError(404, { detail: "Mock: unknown job" })
      if (job.status !== "queued") throw new MockHttpError(409, { detail: "only queued jobs can be edited" })
      const { core, scripts } = body as { core: Record<string, unknown>; scripts: Record<string, Record<string, unknown>> }
      const request = { ...requests.get(job.id)!, core, scripts }
      const errors = validateJob(request)
      if (errors.length > 0) throw new MockHttpError(422, { errors })
      requests.set(job.id, request)
      job.summary = String(core.prompt ?? "").slice(0, 120)
      broadcast({ type: "queue.changed", ...queueStatus() })
      return { ...job, core, scripts, layout_hash: "mock", schema_version: 1 }
    },
  ],
  [
    "POST",
    /^\/outputs\/save$/,
    (_m, body) => {
      const { ids } = body as { ids: string[] }
      // No archive in mock mode: the zip download is skipped.
      return { saved: ids.map((id) => `${id}.png`), zip: null }
    },
  ],
  [
    "POST",
    /^\/jobs\/([^/]+)\/retry$/,
    (m) => {
      const request = requests.get(m[1] ?? "")
      if (!request) throw new MockHttpError(404, { detail: "Mock: unknown job" })
      return enqueueAll({ ...request, repeat: 1 })
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
      broadcast({ type: "queue.changed", ...queueStatus() })
      return undefined
    },
  ],
  [
    "POST",
    /^\/uploads$/,
    async (_m, body) => {
      await delay(250)
      const file = body instanceof FormData ? body.get("file") : null
      const id = `upload-${++uploadSeq}`
      const size = file instanceof Blob ? await imageSize(file) : null
      if (size) imageSizes.set(id, size)
      return { id, name: file instanceof File ? file.name : "upload", width: size?.width ?? null, height: size?.height ?? null }
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
