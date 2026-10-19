/**
 * Mock `/networks` (VITE_MOCK=1 and tests): 5,000 LoRA in nested folders
 * (enough to feel the virtualized grid), embeddings, checkpoints and
 * modules, with the server's search, filters (hidden "." folders shown when
 * searched), sorting, offset / cursor windows, facets, metadata and
 * user-metadata edits. Item 11 has a video preview.
 */
import type { NetworkFolder, NetworkItem, NetworkKind, NetworkMeta } from "../types"

type MockRoute = [method: string, pattern: RegExp, handler: (match: RegExpExecArray, body: unknown, url: URL) => unknown]

const KINDS: NetworkKind[] = [
  { id: "lora", title: "LoRA", action: "prompt", negative: true, fields: ["description", "sd version", "activation text", "preferred weight", "negative text", "tags", "notes"], builtin: true },
  { id: "embeddings", title: "Textual Inversion", action: "prompt", negative: true, fields: ["description", "sd version", "tags", "notes"], builtin: true },
  { id: "checkpoints", title: "Checkpoints", action: "checkpoint", negative: false, fields: ["description", "sd version", "tags", "notes"], builtin: true },
  { id: "vae", title: "VAE", action: "module", negative: false, fields: ["description", "tags", "notes"], builtin: true },
  { id: "text_encoders", title: "Text Encoders", action: "module", negative: false, fields: ["description", "tags", "notes"], builtin: true },
]

const FOLDERS = ["", "styles", "styles/ink", "styles/paint", "characters", "characters/original", "concepts", "anime/girls", "anime/scenery", "poses", "styles/.wip"]
const WORDS = ["watercolor", "ink", "neon", "pastel", "cyber", "retro", "gothic", "sketch", "glow", "dream", "storm", "velvet", "comic", "film", "mecha", "floral"]
const ARCHS = ["anima", "xl", "sd", null, "anima", "flux"] as const
const DAY = 86_400

type Entry = { item: NetworkItem; user: Record<string, unknown> }

function makeItem(kind: string, index: number, name: string, folder: string, extra: Partial<NetworkItem> = {}): NetworkItem {
  const path = folder ? `${folder}/${name}.safetensors` : `${name}.safetensors`
  const arch = ARCHS[index % ARCHS.length] ?? null
  return {
    name,
    title: name,
    kind,
    path,
    folder,
    preview: index % 7 === 3 ? null : `mock-net-${(index * 37) % 360}`,
    preview_type: index % 7 === 3 ? null : index === 11 ? "video" : "image",
    placeholder: null,
    description: index % 5 === 0 ? `A ${WORDS[index % WORDS.length]} look, trained on ${40 + (index % 90)} images.` : null,
    prompt: kind === "lora" ? `<lora:${name}:1>` : name,
    negative_prompt: null,
    activation_text: null,
    preferred_weight: null,
    negative_text: null,
    tags: index % 4 === 0 ? [WORDS[(index + 3) % WORDS.length]!] : [],
    search_terms: [path],
    hash: (index * 2654435761).toString(16).slice(0, 10),
    alias: name,
    sd_version: "Unknown",
    arch,
    subtype: kind === "lora" ? (index % 9 === 0 ? "LyCORIS" : "LoRA") : null,
    value: null,
    mtime: 1_760_000_000 - index * DAY * 0.37,
    created: 1_750_000_000 - ((index * 104_729) % 5000) * DAY * 0.05,
    size: 18_000_000 + ((index * 7919) % 400) * 1_000_000,
    editable: true,
    hidden: folder.split("/").some((part) => part.startsWith(".")),
    ...extra,
  }
}

function build(): Map<string, Entry[]> {
  const lora: Entry[] = []
  for (let i = 0; i < 5000; i++) {
    const name = `${WORDS[i % WORDS.length]}_${WORDS[(i * 7 + 3) % WORDS.length]}_v${(i % 5) + 1}_${i}`
    lora.push({ item: makeItem("lora", i, name, FOLDERS[i % FOLDERS.length]!), user: {} })
  }
  const embeddings = ["easynegative", "bad_hands_5", "ng_deepnegative", "verybadimagenegative", "anima_quality", "detail_tweaker"].map((name, i) => ({
    item: makeItem("embeddings", i, name, i % 2 ? "negative" : "", { subtype: null }),
    user: {},
  }))
  const checkpoints = ["anima-preview-2", "animagine-xl-4.0", "noobai-xl-1.1", "flux1-dev-fp8", "sd15-dreamshaper-8"].map((name, i) => ({
    item: makeItem("checkpoints", i, name, i > 2 ? "other" : "", { value: `${name}.safetensors [${(i * 99991).toString(16).padStart(10, "0")}]`, arch: ["anima", "xl", "xl", "flux", "sd"][i]!, size: 6_400_000_000 }),
    user: {},
  }))
  const vae = ["qwen_image_vae.safetensors", "sdxl_vae.safetensors", "ae.safetensors"].map((name, i) => ({ item: makeItem("vae", i, name, "", { title: name.replace(/\.\w+$/, ""), value: name, arch: null }), user: {} }))
  const encoders = ["qwen_3_06b_base.safetensors", "t5xxl_fp8.safetensors"].map((name, i) => ({ item: makeItem("text_encoders", i, name, "", { title: name.replace(/\.\w+$/, ""), value: name, arch: null }), user: {} }))
  return new Map([
    ["lora", lora],
    ["embeddings", embeddings],
    ["checkpoints", checkpoints],
    ["vae", vae],
    ["text_encoders", encoders],
  ])
}

let data: Map<string, Entry[]> | null = null
const store = () => (data ??= build())

/** The `<lora:name:…>` names of a prompt that no LoRA answers to (like the server, by name or alias). */
export function missingLoras(prompt: string): string[] {
  const known = new Set(store().get("lora")!.flatMap(({ item }) => [item.name, item.alias]))
  const names = [...prompt.matchAll(/<lora:([^:>]+)/g)].map((match) => match[1]!.trim())
  return [...new Set(names.filter((name) => !known.has(name)))]
}

/** Resets the mock (tests). */
export function resetMockNetworks() {
  data = null
}

const currentArch = "anima"

function entriesOf(kind: string): Entry[] {
  const entries = store().get(kind)
  if (!entries) throw new Error(`unknown network kind '${kind}'`)
  return entries
}

function view(entry: Entry): NetworkItem {
  const user = entry.user
  const weight = Number(user["preferred weight"]) || null
  const activation = typeof user["activation text"] === "string" ? user["activation text"] : null
  const item = { ...entry.item }
  if (item.kind === "lora") item.prompt = `<lora:${item.name}:${weight ?? 1}>${activation ? ` ${activation}` : ""}`
  item.preferred_weight = weight
  item.activation_text = activation
  item.negative_text = typeof user["negative text"] === "string" ? user["negative text"] : null
  item.negative_prompt = item.kind === "lora" ? item.negative_text : null
  if (typeof user.description === "string") item.description = user.description
  if (Array.isArray(user.tags)) item.tags = user.tags.map(String)
  if (typeof user["sd version"] === "string") {
    item.sd_version = user["sd version"]
    if (user["sd version"] !== "Unknown") item.arch = user["sd version"]
  }
  return item
}

type Filters = { q: string; folder: string; archs: string[]; compatible: boolean }

function filtersOf(url: URL): Filters {
  const archs = url.searchParams.getAll("arch").flatMap((value) => value.split(",")).filter(Boolean)
  return { q: url.searchParams.get("q") ?? "", folder: url.searchParams.get("folder") ?? "", archs, compatible: url.searchParams.get("compatible") === "true" }
}

function match(items: NetworkItem[], filters: Filters, skip?: "folder" | "arch"): NetworkItem[] {
  const terms = filters.q.toLowerCase().split(/\s+/).filter(Boolean)
  // extra_networks_hidden_models = "When searched"
  const showHidden = filters.q.trim().length >= 4
  return items.filter((item) => {
    if (item.hidden && !showHidden) return false
    const hay = [item.name, item.path, item.description ?? "", item.activation_text ?? "", ...item.tags].join("\n").toLowerCase()
    if (terms.some((term) => !hay.includes(term))) return false
    if (skip !== "folder" && filters.folder && !(item.folder === filters.folder || item.folder?.startsWith(`${filters.folder}/`))) return false
    if (skip !== "arch") {
      if (filters.archs.length && !filters.archs.includes(item.arch ?? "unknown")) return false
      if (filters.compatible && item.arch && item.arch !== currentArch) return false
    }
    return true
  })
}

function folderTree(folders: string[]): NetworkFolder[] {
  const counts = new Map<string, number>()
  for (const folder of folders) {
    if (!folder) continue
    const parts = folder.split("/")
    parts.forEach((_, i) => {
      const path = parts.slice(0, i + 1).join("/")
      counts.set(path, (counts.get(path) ?? 0) + 1)
    })
  }
  const nodes = new Map<string, NetworkFolder>()
  const roots: NetworkFolder[] = []
  for (const path of [...counts.keys()].toSorted()) {
    const node: NetworkFolder = { path, name: path.split("/").at(-1)!, count: counts.get(path)!, children: [] }
    nodes.set(path, node)
    const parent = path.includes("/") ? nodes.get(path.slice(0, path.lastIndexOf("/"))) : undefined
    if (parent) parent.children!.push(node)
    else roots.push(node)
  }
  return roots
}

function find(kind: string, encoded: string): Entry {
  const name = decodeURIComponent(encoded)
  const entry = entriesOf(kind).find((candidate) => candidate.item.name === name)
  if (!entry) throw new Error(`no ${kind} named '${name}'`)
  return entry
}

function meta(entry: Entry): NetworkMeta {
  const item = view(entry)
  const tags = WORDS.map((tag, i) => ({ tag, count: Math.max(1, 120 - i * 9 - (item.name.length % 7)) }))
  return {
    ...item,
    metadata: { ss_output_name: item.name, ss_network_dim: "32", ss_network_alpha: "16", ss_base_model_version: item.arch ?? "unknown", ss_tag_frequency: { "10_set": Object.fromEntries(tags.map((t) => [t.tag, t.count])) } },
    user_metadata: entry.user,
    training: { output_name: item.name, dim: 32, alpha: 16, network_type: item.subtype, epochs: 10, steps: 2400, trained_at: "2026-09-01T12:00:00+00:00", resolutions: [{ resolution: "1024x1024", count: 80 }, { resolution: "832x1216", count: 30 }], dataset_size: 110 },
    dataset_tags: item.kind === "lora" ? tags : [],
  }
}

export const networkRoutes: MockRoute[] = [
  ["GET", /^\/networks$/, () => KINDS],
  ["POST", /^\/networks\/refresh$/, () => ({ ok: true })],
  [
    "GET",
    /^\/networks\/([^/]+)$/,
    (m, _body, url) => {
      const items = match(entriesOf(m[1]!).map(view), filtersOf(url))
      const sort = url.searchParams.get("sort") ?? "name"
      const desc = url.searchParams.get("order") === "desc"
      const key = (item: NetworkItem): string | number =>
        sort === "date" ? (item.mtime ?? 0) : sort === "created" ? (item.created ?? 0) : sort === "size" ? (item.size ?? 0) : sort === "path" ? item.path.toLowerCase() : item.title.toLowerCase()
      items.sort((a, b) => {
        const x = key(a)
        const y = key(b)
        return (x < y ? -1 : x > y ? 1 : 0) * (desc ? -1 : 1)
      })
      const offset = Number(url.searchParams.get("offset") ?? url.searchParams.get("cursor") ?? 0) || 0
      const limit = Math.min(500, Number(url.searchParams.get("limit") ?? 100) || 100)
      const page = items.slice(offset, offset + limit)
      return { items: page, next_cursor: offset + limit < items.length ? String(offset + limit) : null, total: items.length }
    },
  ],
  [
    "GET",
    /^\/networks\/([^/]+)\/facets$/,
    (m, _body, url) => {
      const all = entriesOf(m[1]!).map(view)
      const filters = filtersOf(url)
      const counts = new Map<string, number>()
      for (const item of match(all, filters, "arch")) counts.set(item.arch ?? "unknown", (counts.get(item.arch ?? "unknown") ?? 0) + 1)
      return {
        kind: m[1],
        total: match(all, filters).length,
        folder_total: match(all, filters, "folder").length,
        count: all.length,
        folders: folderTree(match(all, filters, "folder").map((item) => item.folder ?? "")),
        archs: [...counts].map(([id, count]) => ({ id, count })).toSorted((a, b) => Number(a.id === "unknown") - Number(b.id === "unknown") || b.count - a.count),
        current_arch: currentArch,
      }
    },
  ],
  ["POST", /^\/networks\/([^/]+)\/refresh$/, () => ({ ok: true })],
  ["GET", /^\/networks\/([^/]+)\/(.+)\/meta$/, (m) => meta(find(m[1]!, m[2]!))],
  [
    "PATCH",
    /^\/networks\/([^/]+)\/(.+)\/meta$/,
    (m, body) => {
      const entry = find(m[1]!, m[2]!)
      const { user_metadata: changes = {}, replace = false } = (body ?? {}) as { user_metadata?: Record<string, unknown>; replace?: boolean }
      const next: Record<string, unknown> = replace ? {} : { ...entry.user }
      for (const [key, value] of Object.entries(changes)) {
        if (value === null) delete next[key]
        else next[key] = value
      }
      entry.user = next
      return meta(entry)
    },
  ],
  [
    "PUT",
    /^\/networks\/([^/]+)\/(.+)\/preview$/,
    (m, body) => {
      const entry = find(m[1]!, m[2]!)
      entry.item = { ...entry.item, preview: String((body as { file_id?: unknown })?.file_id ?? entry.item.preview) }
      return view(entry)
    },
  ],
  ["POST", /^\/networks\/([^/]+)\/(.+)\/open-folder$/, () => ({ ok: true })],
]
