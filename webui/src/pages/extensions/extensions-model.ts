/**
 * Extensions page logic: the extension index (filter by text and tags,
 * sort like the former "Available" tab), git URL checks, and the staged
 * enable/disable/update changes of the installed list.
 */
import type { components } from "@/api/gen/schema"
import { ApiError } from "@/api/http"

type Schemas = components["schemas"]
export type ExtensionInfo = Schemas["ExtensionInfo"]
export type ExtensionList = Schemas["ExtensionList"]
export type IndexEntry = Schemas["IndexEntry"]
export type ExtensionIndex = Schemas["ExtensionIndex"]
export type ConfigStateInfo = Schemas["ConfigStateInfo"]
export type ConfigStateDetail = Schemas["ConfigStateDetail"]

// ---------------------------------------------------------------- errors

export type Translate = (key: string, options?: Record<string, unknown>) => string

/**
 * A server error in the user's language: extension and system errors carry
 * their English `template` (`{{name}}` placeholders, a catalog key) and
 * `params`; anything else is its message (translated when it is a key).
 */
export function adminErrorMessage(error: unknown, t: Translate): string {
  if (error instanceof ApiError && error.detail && typeof error.detail === "object") {
    const { template, params } = error.detail as { template?: unknown; params?: unknown }
    if (typeof template === "string" && template) {
      const values = params && typeof params === "object" && !Array.isArray(params) ? (params as Record<string, unknown>) : {}
      return t(template, { ...values, defaultValue: error.message })
    }
  }
  if (error instanceof Error) return t(error.message, { defaultValue: error.message })
  return String(error)
}

// ---------------------------------------------------------------- URLs

const GIT_URL = /^(https?|ssh|git):\/\/\S+$|^git@[^\s:]+:\S+$/
const DIRNAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$/

export function isGitUrl(url: string): boolean {
  return GIT_URL.test(url.trim())
}

/** `https://github.com/a/sd-ext.git` → `sd-ext` (the server's default folder name). */
export function dirnameFromUrl(url: string): string {
  let clean = url.trim().replace(/\/+$/, "")
  if (clean.endsWith(".git")) clean = clean.slice(0, -4)
  return clean.split("/").pop()?.split(":").pop() ?? ""
}

export function isValidDirname(name: string): boolean {
  return DIRNAME.test(name)
}

// ---------------------------------------------------------------- index

export const SORTS = ["newest", "oldest", "a-z", "z-a", "index", "updated", "created", "stars"] as const
export type IndexSort = (typeof SORTS)[number]

export const SORT_LABELS: Record<IndexSort, string> = {
  newest: "Newest first",
  oldest: "Oldest first",
  "a-z": "Name A–Z",
  "z-a": "Name Z–A",
  index: "Index order",
  updated: "Recently updated",
  created: "Recently created",
  stars: "Most stars",
}

/** Tags hidden by default (as the former tab did). */
export const DEFAULT_HIDDEN_TAGS = ["ads", "localization", "installed"]

export type IndexFilter = {
  query: string
  tags: ReadonlySet<string>
  /** `hide`: drop entries having the selected tags; `show`: keep only those. */
  tagMode: "hide" | "show"
  /** Whether an entry needs any or all of the selected tags to count as having them. */
  match: "any" | "all"
  sort: IndexSort
}

/** An entry's tags, with `installed` added when it is. */
export function entryTags(entry: IndexEntry): string[] {
  const tags = entry.tags ?? []
  return entry.installed && !tags.includes("installed") ? [...tags, "installed"] : tags
}

function hasTags(entry: IndexEntry, selected: ReadonlySet<string>, match: "any" | "all"): boolean {
  const tags = entryTags(entry)
  const found = tags.filter((tag) => selected.has(tag)).length
  return match === "any" ? found > 0 : found === selected.size
}

const text = (value: string | null | undefined) => value ?? ""

function compare(sort: IndexSort): (a: IndexEntry, b: IndexEntry) => number {
  const by = <T>(key: (e: IndexEntry) => T, reverse: boolean) => (a: IndexEntry, b: IndexEntry) => {
    const x = key(a)
    const y = key(b)
    const order = x < y ? -1 : x > y ? 1 : 0
    return (reverse ? -order : order) || (a.order ?? 0) - (b.order ?? 0)
  }
  /** Dated entries in date order; undated ones after them, in index order. */
  const byDate = (key: (e: IndexEntry) => string | null | undefined, reverse: boolean) => {
    const dated = by((e) => text(key(e)), reverse)
    return (a: IndexEntry, b: IndexEntry) => {
      const x = Boolean(key(a))
      const y = Boolean(key(b))
      if (x !== y) return x ? -1 : 1
      return x ? dated(a, b) : (a.order ?? 0) - (b.order ?? 0)
    }
  }
  switch (sort) {
    case "newest":
      return byDate((e) => e.added, true)
    case "oldest":
      return byDate((e) => e.added, false)
    case "a-z":
      return by((e) => e.name.toLowerCase(), false)
    case "z-a":
      return by((e) => e.name.toLowerCase(), true)
    case "updated":
      return byDate((e) => e.commit_time, true)
    case "created":
      return byDate((e) => e.created_at, true)
    case "stars":
      return by((e) => e.stars ?? 0, true)
    default:
      return (a, b) => (a.order ?? 0) - (b.order ?? 0)
  }
}

/** Entries passing the text and tag filters, sorted. */
export function filterIndex(items: readonly IndexEntry[], filter: IndexFilter): IndexEntry[] {
  const terms = filter.query.toLowerCase().split(/\s+/).filter(Boolean)
  const out = items.filter((entry) => {
    if (filter.tags.size > 0) {
      const has = hasTags(entry, filter.tags, filter.match)
      if (filter.tagMode === "hide" ? has : !has) return false
    }
    if (terms.length === 0) return true
    const haystack = `${entry.name}\n${text(entry.description)}\n${entry.url}\n${entryTags(entry).join(" ")}`.toLowerCase()
    return terms.every((term) => haystack.includes(term))
  })
  return out.sort(compare(filter.sort))
}

/** "2023-04-01T10:00:00Z" → a local date; "" for missing or unreadable values. */
export function formatDate(value: string | null | undefined, locale?: string): string {
  if (!value) return ""
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString(locale, { year: "numeric", month: "short", day: "numeric" })
}

// ---------------------------------------------------------------- staged changes

export type Staged = {
  /** name → enabled, for extensions whose switch differs from the server */
  enabled: Record<string, boolean>
  /** names marked for update */
  update: string[]
}

export const NO_CHANGES: Staged = { enabled: {}, update: [] }

export function isEnabled(ext: ExtensionInfo, staged: Staged): boolean {
  return Object.hasOwn(staged.enabled, ext.name) ? staged.enabled[ext.name]! : ext.enabled
}

/** Toggles an extension's switch; a switch back to the server's value is no change. */
export function stageEnabled(staged: Staged, ext: ExtensionInfo, enabled: boolean): Staged {
  const next = { ...staged.enabled }
  if (enabled === ext.enabled) delete next[ext.name]
  else next[ext.name] = enabled
  return { ...staged, enabled: next }
}

export function stageUpdate(staged: Staged, name: string, update: boolean): Staged {
  const without = staged.update.filter((item) => item !== name)
  return { ...staged, update: update ? [...without, name] : without }
}

export function stagedCount(staged: Staged): number {
  return Object.keys(staged.enabled).length + staged.update.length
}

/** The `disabled` list for `PATCH /extensions` after the staged switches (null: unchanged). */
export function disabledList(items: readonly ExtensionInfo[], staged: Staged): string[] | null {
  if (Object.keys(staged.enabled).length === 0) return null
  return items.filter((ext) => !isEnabled(ext, staged)).map((ext) => ext.name)
}

/** Drops staged changes that no longer apply (extensions removed, updates no longer available, switches now matching). */
export function pruneStaged(staged: Staged, items: readonly ExtensionInfo[]): Staged {
  const byName = new Map(items.map((ext) => [ext.name, ext]))
  const enabled: Record<string, boolean> = {}
  for (const [name, value] of Object.entries(staged.enabled)) {
    const ext = byName.get(name)
    if (ext && ext.enabled !== value) enabled[name] = value
  }
  const update = staged.update.filter((name) => byName.get(name)?.can_update)
  return { enabled, update }
}
