/**
 * Mock routes of the settings and extensions pages: the real settings schema
 * (contract/settings.json), installed extensions with update checks, a large
 * synthetic extension index, config-state backups, licenses, sysinfo, server
 * status, and a "Reload UI" / restart during which /internal/ping fails for
 * a few seconds.
 */
import type { SettingsSchema } from "@/ui/schema"
import settingsContract from "./contract/settings.json"
import { bootstrap } from "./fixtures"
import { MockHttpError } from "./server"
import { broadcast } from "./socket"

type Handler = (match: RegExpExecArray, body: unknown, url: URL) => unknown | Promise<unknown>
const QUICK = "quicksettings_list"
const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

// ---------------------------------------------------------------- settings

const settings = settingsContract as unknown as SettingsSchema
const values: Record<string, unknown> = { ...settings.values, ...bootstrap.options }

function schema(): SettingsSchema {
  return { ...settings, quicksettings: (values.quicksettings_list as string[]) ?? [], values: values as SettingsSchema["values"] }
}

function patchOptions(body: unknown) {
  const changes = (body ?? {}) as Record<string, unknown>
  // Try "#invalid" in a text setting to see a rejected value.
  const errors = Object.entries(changes).flatMap(([key, value]) => {
    if (!(key in values)) return [{ owner: "settings", param: key, message: "unknown setting" }]
    if (typeof value === "string" && value.includes("#invalid")) return [{ owner: "settings", param: key, message: "Mock: rejected value" }]
    return []
  })
  if (errors.length > 0) throw new MockHttpError(422, { errors })
  return changes
}

// ---------------------------------------------------------------- extensions

type Ext = {
  name: string
  enabled: boolean
  builtin: boolean
  remote: string | null
  remote_url: string | null
  branch: string | null
  version: string | null
  commit_hash: string | null
  commit_url: string | null
  commit_date: string | null
  status: string | null
  can_update: boolean
}

const BUILTIN = ["sd_forge_controlnet", "sd_forge_lora", "sd-webui-tagcomplete-neo", "agent-scheduler", "soft-inpainting", "sd_forge_anima_presets", "mobile", "sd-webui-detail-daemon"]
const USER = ["sd-webui-infinite-image-browsing", "stable-diffusion-webui-wd14-tagger", "sd-dynamic-prompts", "adetailer-extra"]

function hash(name: string): string {
  let h = 2166136261
  for (const c of name) h = Math.imul(h ^ c.charCodeAt(0), 16777619)
  return (h >>> 0).toString(16).padStart(8, "0").repeat(5)
}

const extensions: Ext[] = [
  ...USER.map((name, index): Ext => {
    const remote = `https://github.com/example/${name}`
    const commit = hash(name)
    return { name, enabled: index !== 2, builtin: false, remote, remote_url: remote, branch: "main", version: commit.slice(0, 8), commit_hash: commit, commit_url: `${remote}/commit/${commit}`, commit_date: new Date(Date.UTC(2026, 6, 1 + index * 9)).toISOString(), status: null, can_update: false }
  }),
  ...BUILTIN.map((name): Ext => ({ name, enabled: true, builtin: true, remote: null, remote_url: null, branch: null, version: null, commit_hash: null, commit_url: null, commit_date: null, status: null, can_update: false })),
]
let disableAll = "none"

function listing() {
  return { items: extensions, disable_all: disableAll, access: true, restartable: true, disabled_by: disableAll === "none" ? null : disableAll, index_url: "https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui-extensions/master/index.json" }
}

function find(name: string): Ext {
  const ext = extensions.find((item) => item.name === name)
  if (!ext) throw new MockHttpError(404, { detail: `no extension ${name}`, template: "no extension {{name}}", params: { name } })
  return ext
}

const TAGS = ["script", "tab", "editing", "manipulations", "online", "animation", "query", "training", "models", "UI related", "extras", "prompting", "ads", "localization"]
const WORDS = ["prompt", "tagger", "browser", "upscale", "mask", "pose", "lora", "style", "grid", "video", "depth", "color", "face", "batch", "wildcard", "inpaint"]

const index = Array.from({ length: 1600 }, (_, i) => {
  const name = `${WORDS[i % WORDS.length]}-${WORDS[(i * 7) % WORDS.length]}-${i}`
  const day = new Date(Date.UTC(2022, 0, 1) + i * 86_400_000).toISOString()
  return {
    name: i === 3 ? USER[0]! : name,
    url: i === 3 ? `https://github.com/example/${USER[0]}` : `https://github.com/mock/${name}`,
    description: `Adds ${WORDS[(i * 3) % WORDS.length]} tools to the ${WORDS[(i * 5) % WORDS.length]} workflow.`,
    tags: [TAGS[i % TAGS.length]!, TAGS[(i * 11) % TAGS.length]!].filter((tag, n, all) => all.indexOf(tag) === n),
    added: day.slice(0, 10),
    created_at: day,
    commit_time: new Date(Date.UTC(2026, 0, 1) - (i % 400) * 86_400_000).toISOString(),
    stars: (i * 37) % 2400,
  }
})

function indexResponse(url: string) {
  const names = new Set(extensions.map((ext) => ext.name))
  return {
    url,
    fetched_at: new Date().toISOString(),
    items: index.map((entry, order) => ({ ...entry, order, installed: names.has(entry.url.split("/").pop() ?? "") })),
    tags: { ...Object.fromEntries(TAGS.map((tag) => [tag, `${tag} extensions`])), installed: "extensions that are already installed" },
  }
}

type State = { id: string; name: string; created_at: string; commits: Record<string, string | null> }
const states: State[] = [
  { id: "2026_09_30-12_00_00_Backup (pre-update).json", name: "Backup (pre-update)", created_at: "2026-09-30T12:00:00Z", commits: Object.fromEntries(extensions.map((ext) => [ext.name, ext.commit_hash ? hash(`${ext.name}-old`) : null])) },
]

function stateDetail(state: State) {
  const repo = (commit: string | null, remote: string | null) => ({ remote, branch: remote ? "main" : null, commit_hash: commit, commit_url: commit && remote ? `${remote}/commit/${commit}` : null, commit_date: null })
  return {
    id: state.id,
    name: state.name,
    created_at: state.created_at,
    webui_commit: "a1b2c3d4e5",
    extensions: Object.keys(state.commits).length,
    webui: { saved: repo("a1b2c3d4e5", "https://github.com/example/remi"), current: repo("f6e7d8c9b0", "https://github.com/example/remi") },
    items: extensions.map((ext) => ({
      name: ext.name,
      builtin: ext.builtin,
      saved: ext.name in state.commits ? { ...repo(state.commits[ext.name] ?? null, ext.remote), enabled: true } : null,
      current: { ...repo(ext.commit_hash, ext.remote), enabled: ext.enabled },
    })),
  }
}

// ---------------------------------------------------------------- server

let downUntil = 0
/** Changes when the mock server "goes down" (a reload or restart). */
let instance = "mock-1"
let hashes = { running: false, done: 0, total: 0, failed: [] as string[], current: [] as string[], finished_at: null as string | null }
let modelLoaded: string | null = String(bootstrap.options.sd_model_checkpoint ?? "") || null

const LICENSES = {
  groups: [
    { title: "Special Thanks", items: [{ id: "comfy", name: "ComfyUI", url: "https://github.com/Comfy-Org/ComfyUI", description: "Many inference code and structural optimizations were taken from this repository", license_url: "https://raw.githubusercontent.com/Comfy-Org/ComfyUI/master/LICENSE" }] },
    {
      title: "Diffusion Engine",
      items: [
        { id: "sd1", name: "Stable Diffusion", url: "https://github.com/Stability-AI/stablediffusion", description: "Inference Code for SD1", license_url: "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/LICENSE" },
        { id: "flux", name: "Flux", url: "https://github.com/black-forest-labs/flux", description: "Reference Code for Flux", license_url: "https://raw.githubusercontent.com/black-forest-labs/flux/main/LICENSE" },
      ],
    },
  ],
}

export const adminRoutes: [method: string, pattern: RegExp, handler: Handler][] = [
  ["GET", /^\/options$/, () => schema()],
  [
    "PATCH",
    /^\/options$/,
    async (_m, body) => {
      await delay(350)
      const changes = patchOptions(body)
      Object.assign(values, changes)
      for (const key of Object.keys(changes)) if (key in bootstrap.options) (bootstrap.options as Record<string, unknown>)[key] = changes[key]
      if (QUICK in changes) (bootstrap.options as Record<string, unknown>)[QUICK] = changes[QUICK]
      broadcast({ type: "options.changed", keys: Object.keys(changes) })
      return schema()
    },
  ],
  [
    "POST",
    /^\/options\/validate$/,
    (_m, body) => {
      patchOptions(body)
      return { ok: true }
    },
  ],
  ["GET", /^\/internal\/ping$/, () => {
    if (Date.now() < downUntil) throw new MockHttpError(503, { detail: "Mock: restarting" })
    return { instance }
  }],
  ["GET", /^\/server\/status$/, () => ({ restartable: true, extension_access: true, settings_frozen: false, model_loaded: modelLoaded, version: "mock" })],
  [
    "POST",
    /^\/server\/(reload|restart)$/,
    (m) => {
      setTimeout(() => {
        downUntil = Date.now() + (m[1] === "restart" ? 5000 : 3000)
        instance = `mock-${Date.now()}`
      }, 1200)
      return { ok: true, message: m[1] === "restart" ? "restarting" : "reloading" }
    },
  ],
  ["POST", /^\/models\/unload$/, () => ((modelLoaded = null), { ok: true })],
  [
    "POST",
    /^\/models\/reload$/,
    async () => {
      broadcast({ type: "model.loading", stage: "loading", model: bootstrap.options.sd_model_checkpoint })
      await delay(1200)
      modelLoaded = String(bootstrap.options.sd_model_checkpoint ?? "model")
      broadcast({ type: "model.loaded", model: modelLoaded })
      return { ok: true, message: modelLoaded }
    },
  ],
  [
    "GET",
    /^\/models\/loaded$/,
    () => ({
      checkpoint: modelLoaded,
      items: modelLoaded
        ? [
            { name: "KModel", memory_mb: 4870, device: "mps" },
            { name: "IntegratedAutoencoderKL", memory_mb: 160, device: "mps" },
            { name: "Qwen3Model", memory_mb: 1190, device: "cpu" },
          ]
        : [],
    }),
  ],
  ["GET", /^\/models\/hashes$/, () => hashes],
  [
    "POST",
    /^\/models\/hashes$/,
    () => {
      if (hashes.running) throw new MockHttpError(409, { detail: "the checkpoints are being hashed already" })
      const total = 6
      hashes = { running: true, done: 0, total, failed: [], current: [], finished_at: null }
      const tick = setInterval(() => {
        hashes = { ...hashes, done: hashes.done + 1 }
        if (hashes.done >= total) {
          clearInterval(tick)
          hashes = { ...hashes, running: false, finished_at: new Date().toISOString() }
        }
      }, 700)
      return hashes
    },
  ],
  ["GET", /^\/licenses$/, () => LICENSES],
  [
    "GET",
    /^\/sysinfo$/,
    () => ({
      Platform: "macOS-27.0-arm64",
      Python: "3.13.1",
      Version: "remi-mock",
      "Git status": "On branch remi-webui",
      "Script path": "/opt/remi",
      "Data path": "/opt/remi",
      "Extensions dir": "/opt/remi/extensions",
      Checksum: "0".repeat(64),
      Commandline: ["launch.py", "--listen"],
      CPU: { model: "Mock CPU", count_logical: 12 },
      RAM: { total: "32GB", used: "12GB" },
      Extensions: extensions.filter((ext) => ext.enabled).map((ext) => ({ name: ext.name, path: ext.name, commit: ext.commit_hash, branch: ext.branch, remote: ext.remote })),
      Config: { samples_format: "png" },
      Packages: ["torch==2.14.1", "fastapi==0.120.0"],
    }),
  ],
  ["POST", /^\/sysinfo\/check$/, (_m, body) => ({ valid: String((body as { text?: string }).text ?? "").includes('"Checksum"') })],
  ["GET", /^\/extensions$/, () => listing()],
  [
    "PATCH",
    /^\/extensions$/,
    async (_m, body) => {
      await delay(250)
      const { disabled, disable_all } = body as { disabled?: string[]; disable_all?: string }
      if (disabled) for (const ext of extensions) ext.enabled = !disabled.includes(ext.name)
      if (disable_all) disableAll = disable_all
      return listing()
    },
  ],
  [
    "POST",
    /^\/extensions\/([^/]+)\/check$/,
    async (m) => {
      const ext = find(decodeURIComponent(m[1] ?? ""))
      await delay(300 + Math.random() * 900)
      if (ext.name.includes("tagger")) {
        ext.status = "unknown (remote error)"
        ext.can_update = false
      } else {
        ext.can_update = ext.name.length % 2 === 0
        ext.status = ext.can_update ? "new commits" : "latest"
      }
      return ext
    },
  ],
  [
    "POST",
    /^\/extensions\/update$/,
    async (_m, body) => {
      await delay(900)
      for (const name of (body as { names: string[] }).names) {
        const ext = find(name)
        ext.commit_hash = hash(`${name}-${Date.now()}`)
        ext.version = ext.commit_hash.slice(0, 8)
        ext.can_update = false
        ext.status = "latest"
      }
      return listing()
    },
  ],
  [
    "POST",
    /^\/extensions\/install$/,
    async (_m, body) => {
      await delay(1500)
      const { url, dirname } = body as { url: string; dirname?: string | null }
      const name = dirname || url.replace(/\/+$/, "").replace(/\.git$/, "").split("/").pop() || "extension"
      if (extensions.some((ext) => ext.name === name)) throw new MockHttpError(409, { detail: `the folder ${name} already exists`, template: "the folder {{name}} already exists", params: { name } })
      if (!/^(https?|ssh|git):\/\/|^git@/.test(url)) throw new MockHttpError(422, { detail: "expected a git URL (https://, ssh://, git:// or git@host:path)", template: "expected a git URL (https://, ssh://, git:// or git@host:path)", params: {} })
      const commit = hash(name)
      const ext: Ext = { name, enabled: true, builtin: false, remote: url, remote_url: url, branch: "main", version: commit.slice(0, 8), commit_hash: commit, commit_url: null, commit_date: new Date().toISOString(), status: null, can_update: false }
      extensions.unshift(ext)
      return ext
    },
  ],
  [
    "DELETE",
    /^\/extensions\/([^/]+)$/,
    (m) => {
      const ext = find(decodeURIComponent(m[1] ?? ""))
      if (ext.builtin) throw new MockHttpError(422, { detail: "built-in extensions cannot be uninstalled; disable them instead", template: "built-in extensions cannot be uninstalled; disable them instead", params: {} })
      extensions.splice(extensions.indexOf(ext), 1)
      return undefined
    },
  ],
  [
    "GET",
    /^\/extensions\/index$/,
    async (_m, _body, url) => {
      await delay(700)
      return indexResponse(url.searchParams.get("url") || listing().index_url)
    },
  ],
  ["GET", /^\/extensions\/config-states$/, () => states.map((state) => ({ id: state.id, name: state.name, created_at: state.created_at, webui_commit: "a1b2c3d4e5", extensions: Object.keys(state.commits).length }))],
  [
    "POST",
    /^\/extensions\/config-states$/,
    (_m, body) => {
      const name = String((body as { name?: string }).name || "Config")
      const now = new Date()
      const state: State = { id: `${now.toISOString().replace(/[:T]/g, "_").slice(0, 19)}_${name}.json`, name, created_at: now.toISOString(), commits: Object.fromEntries(extensions.map((ext) => [ext.name, ext.commit_hash])) }
      states.unshift(state)
      return { id: state.id, name, created_at: state.created_at, webui_commit: "f6e7d8c9b0", extensions: extensions.length }
    },
  ],
  [
    "GET",
    /^\/extensions\/config-states\/([^/]+)$/,
    (m) => {
      const state = states.find((item) => item.id === decodeURIComponent(m[1] ?? ""))
      if (!state) throw new MockHttpError(404, { detail: "no config state" })
      return stateDetail(state)
    },
  ],
  [
    "DELETE",
    /^\/extensions\/config-states\/([^/]+)$/,
    (m) => {
      const at = states.findIndex((item) => item.id === decodeURIComponent(m[1] ?? ""))
      if (at < 0) throw new MockHttpError(404, { detail: "no config state" })
      states.splice(at, 1)
      return undefined
    },
  ],
  [
    "POST",
    /^\/extensions\/config-states\/([^/]+)\/restore$/,
    (_m, body) => {
      const what = (body as { what?: string }).what ?? "extensions"
      if (what !== "extensions") throw new MockHttpError(422, { detail: "only the extensions of a backup can be restored; the WebUI checkout is never reset from here" })
      return { reload: true }
    },
  ],
]
