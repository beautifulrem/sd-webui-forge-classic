/**
 * Mock routes of the Gallery, PNG Info and the Queue tools (VITE_MOCK=1):
 * a generated library of outputs with varied sizes, models and dates,
 * stars, removal, zip, reveal, PNG Info, and the importer with a sample
 * report. Images are the dev server's SVG artwork (`mock-*` ids).
 */
import type { GalleryItem, GalleryQuery, ImageMetadata, ImportReport, ImportSourceInfo } from "../types"
import { MockHttpError } from "./server"
import { broadcast } from "./socket"

type Handler = (match: RegExpExecArray, body: unknown, url: URL) => unknown | Promise<unknown>


const MODELS = ["anima-v1.safetensors", "sdxl-base.safetensors", "illustrious-xl.safetensors"]
const SIZES: [number, number][] = [
  [1024, 1024],
  [832, 1216],
  [1216, 832],
  [768, 1344],
  [1344, 768],
  [2048, 2048],
]
const CATEGORIES = ["txt2img", "txt2img", "txt2img", "img2img", "extras", "grids"] as const
const PROMPTS = ["a red fox in fresh snow", "lighthouse at dusk, volumetric light", "portrait of a knight, intricate armor", "city street in the rain, neon", "watercolor garden with koi pond"]

const now = Date.now()
// VITE_MOCK_GALLERY_SIZE=10000 gives the scrolling measurement its 10,000 images (docs/research/ui-performance.md)
const library: GalleryItem[] = Array.from({ length: Number(import.meta.env.VITE_MOCK_GALLERY_SIZE) || 260 }, (_, index) => {
  const [width, height] = SIZES[index % SIZES.length]!
  const category = CATEGORIES[index % CATEGORIES.length]!
  const model = MODELS[index % MODELS.length]!
  const prompt = PROMPTS[index % PROMPTS.length]!
  return {
    id: `mock-lib${index}-${(index * 37) % 360}`,
    name: `${String(index).padStart(5, "0")}-${index * 7919}.png`,
    width: category === "grids" ? width * 2 : width,
    height,
    size: 900_000 + ((index * 7717) % 600_000),
    mime: "image/png",
    created_at: new Date(now - index * 3_600_000 * 5).toISOString(),
    job_id: null,
    tab: category === "grids" ? "txt2img" : category,
    category,
    model,
    starred: index % 11 === 0,
    infotext: `${prompt}\nNegative prompt: lowres, blurry\nSteps: 28, Sampler: Euler a, Schedule type: Karras, CFG scale: 5, Seed: ${100000 + index}, Size: ${width}x${height}, Model: ${model}`,
  }
})

/** Bumped by every change to the library (stars, removal): the filtered list is cached per query until then. */
let libraryVersion = 0
let lastFiltered: { key: string; version: number; items: GalleryItem[] } | null = null

/**
 * The query's items, cached: a page of a large mock library (VITE_MOCK_GALLERY_SIZE) must not
 * filter and sort every image on the page's main thread, as no real server does.
 */
function filteredCached(query: GalleryQuery): GalleryItem[] {
  const key = JSON.stringify(query)
  if (lastFiltered?.key !== key || lastFiltered.version !== libraryVersion) lastFiltered = { key, version: libraryVersion, items: filtered(query) }
  return lastFiltered.items
}

function filtered(query: GalleryQuery): GalleryItem[] {
  const terms = (query.q ?? "").toLowerCase().split(/\s+/).filter(Boolean)
  const since = query.since ? Date.parse(query.since) : null
  const until = query.until ? Date.parse(query.until) : null
  const items = library.filter((item) => {
    const time = Date.parse(item.created_at)
    const text = `${item.infotext ?? ""} ${item.name} ${item.model ?? ""}`.toLowerCase()
    return (
      (!query.category || item.category === query.category) &&
      (!query.model || item.model === query.model) &&
      (since === null || time >= since) &&
      (until === null || time < until) &&
      (query.starred === undefined || query.starred === null || item.starred === query.starred) &&
      terms.every((term) => text.includes(term))
    )
  })
  const sort = query.sort ?? "newest"
  const by = {
    newest: (a: GalleryItem, b: GalleryItem) => Date.parse(b.created_at) - Date.parse(a.created_at),
    oldest: (a: GalleryItem, b: GalleryItem) => Date.parse(a.created_at) - Date.parse(b.created_at),
    name: (a: GalleryItem, b: GalleryItem) => a.name.localeCompare(b.name),
    largest: (a: GalleryItem, b: GalleryItem) => (b.width ?? 0) * (b.height ?? 0) - (a.width ?? 0) * (a.height ?? 0),
  }[sort]
  return items.sort(by)
}

function queryOf(url: URL): GalleryQuery {
  const get = (key: string) => url.searchParams.get(key) ?? undefined
  const starred = get("starred")
  return {
    q: get("q"),
    category: get("category") as GalleryQuery["category"],
    model: get("model"),
    since: get("since"),
    until: get("until"),
    starred: starred === undefined ? undefined : starred === "true",
    sort: (get("sort") as GalleryQuery["sort"]) ?? "newest",
  }
}

const ids = (body: unknown) => new Set(((body as { ids?: string[] }).ids ?? []).map(String))

const metadata = (item: GalleryItem | undefined, id: string): ImageMetadata => ({
  name: item?.name ?? `${id}.png`,
  format: "PNG",
  mode: "RGB",
  width: item?.width ?? 1024,
  height: item?.height ?? 1024,
  size: item?.size ?? 1_048_576,
  infotext: item?.infotext ?? null,
  params: item?.infotext
    ? { Prompt: item.infotext.split("\n")[0]!, "Negative prompt": "lowres, blurry", Steps: "28", Sampler: "Euler a", "CFG scale": "5", Model: item.model ?? "" }
    : {},
  entries: [
    { group: "Image", key: "Format", value: "PNG", kind: "text" },
    { group: "Image", key: "Size", value: `${item?.width ?? 1024} × ${item?.height ?? 1024}`, kind: "text" },
    { group: "Metadata", key: "Software", value: "Remi WebUI (mock)", kind: "text" },
    { group: "Metadata", key: "ICC profile", value: "<3144 bytes>", kind: "binary" },
  ],
  chunks: [
    { type: "IHDR", length: 13, count: 1 },
    { type: "iCCP", length: 3144, count: 1 },
    { type: "tEXt", length: 412, count: 1, key: "parameters" },
    { type: "IDAT", length: 1_040_384, count: 127 },
    { type: "IEND", length: 0, count: 1 },
  ],
})

const sources: ImportSourceInfo[] = [
  { id: "agent-scheduler", label: "Agent Scheduler", available: true, count: 14, path: "/data/extension-data/agent-scheduler/task_scheduler.sqlite3" },
  { id: "prompt-queue", label: "Prompt Queue", available: false, count: 0, path: null },
]

const report: ImportReport = {
  source: "agent-scheduler",
  path: sources[0]!.path!,
  total: 14,
  queued: 3,
  history: 9,
  refused: 1,
  invalid: 1,
  skipped: 0,
  errors: 0,
  paused: true,
  items: [
    { source_id: "t-1", title: "Night fox", outcome: "queued", notes: [] },
    { source_id: "t-2", title: "castle on a cliff", outcome: "refused", notes: [{ message: "Script arguments past the current parameters could not be mapped: {{count}}.", values: { count: 2 } }] },
    { source_id: "t-3", title: "repaint", outcome: "invalid", notes: [{ message: "Some values do not fit the current parameters; edit the job to queue it.", values: {} }] },
  ],
}

export const libraryRoutes: [method: string, pattern: RegExp, handler: Handler][] = [
  [
    "GET",
    /^\/gallery$/,
    (_m, _body, url) => {
      const items = filteredCached(queryOf(url))
      const start = Number(url.searchParams.get("cursor") ?? 0) || 0
      const limit = Number(url.searchParams.get("limit") ?? 100) || 100
      const page = items.slice(start, start + limit)
      return { items: page, next_cursor: start + limit < items.length ? String(start + limit) : null, total: items.length }
    },
  ],
  [
    "GET",
    /^\/gallery\/facets$/,
    () => {
      const count = (pick: (item: GalleryItem) => string | null | undefined) => {
        const counts = new Map<string, number>()
        for (const item of library) {
          const key = pick(item)
          if (key) counts.set(key, (counts.get(key) ?? 0) + 1)
        }
        return counts
      }
      const models = count((item) => item.model)
      const categories = count((item) => item.category)
      const times = library.map((item) => item.created_at).sort()
      return {
        models: [...models].map(([name, n]) => ({ name, count: n })).sort((a, b) => b.count - a.count),
        categories: ["txt2img", "img2img", "extras", "grids", "saved", "other"].map((name) => ({ name, count: categories.get(name) ?? 0 })),
        oldest: times[0] ?? null,
        newest: times.at(-1) ?? null,
        total: library.length,
        starred: library.filter((item) => item.starred).length,
      }
    },
  ],
  [
    "POST",
    /^\/gallery\/star$/,
    (_m, body) => {
      const wanted = ids(body)
      const starred = (body as { starred?: boolean }).starred ?? true
      let count = 0
      for (const item of library) {
        if (!wanted.has(item.id)) continue
        item.starred = starred
        count++
      }
      libraryVersion++
      broadcast({ type: "outputs.changed", ids: [...wanted] })
      return { count }
    },
  ],
  [
    "POST",
    /^\/gallery\/delete$/,
    (_m, body) => {
      const wanted = ids(body)
      const before = library.length
      library.splice(0, library.length, ...library.filter((item) => !wanted.has(item.id)))
      libraryVersion++
      broadcast({ type: "outputs.changed", ids: [...wanted] })
      return { count: before - library.length }
    },
  ],
  ["POST", /^\/gallery\/zip$/, (_m, body) => ({ id: "mock-zip-0", name: "gallery-mock.zip", count: ids(body).size })],
  ["POST", /^\/gallery\/([^/]+)\/reveal$/, () => ({ ok: true })],
  ["POST", /^\/outputs\/refresh$/, () => ({ added: 0 })],
  [
    "GET",
    /^\/gallery\/([^/]+)$/,
    (m) => {
      const item = library.find((entry) => entry.id === decodeURIComponent(m[1] ?? ""))
      if (!item) throw new MockHttpError(404, { detail: "no such output" })
      return item
    },
  ],
  [
    "POST",
    /^\/pnginfo$/,
    (_m, body) => {
      const id = String((body as { file_id?: unknown }).file_id ?? "")
      return metadata(
        library.find((item) => item.id === id),
        id,
      )
    },
  ],
  ["GET", /^\/queue\/import$/, () => sources],
  ["POST", /^\/queue\/import\/(agent-scheduler|prompt-queue)$/, () => report],
  ["POST", /^\/jobs\/star$/, (_m, body) => ({ count: ids(body).size })],
  ["POST", /^\/jobs\/clear-history$/, () => ({ count: 0 })],
]

