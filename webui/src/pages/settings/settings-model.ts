/**
 * The settings page's pure logic: the schema (schema-format.md §6) indexed
 * into sections and items, search over labels / help / keys, the "changed"
 * and "unsaved" filters, and the flat row list the virtualized view draws.
 */
import { jsonEqual } from "@/ui/conditions"
import type { HtmlNode, Json, Node, RowNode, SeparatorNode, SettingNode, SettingsSchema } from "@/ui/schema"
import { isInput } from "@/ui/schema"

export type SettingItem =
  | { kind: "setting"; key: string; node: SettingNode }
  /** Settings shown side by side (a former `OptionRow` group). */
  | { kind: "row"; keys: string[]; node: RowNode; nodes: SettingNode[] }
  | { kind: "html"; node: HtmlNode }
  | { kind: "separator"; node: SeparatorNode }

export type SettingsSection = {
  id: string
  label: string
  categoryId: string
  categoryLabel: string
  items: SettingItem[]
}

export type SettingsCategory = { id: string; label: string; sections: SettingsSection[] }

export type SettingsIndex = {
  categories: SettingsCategory[]
  sections: SettingsSection[]
  /** key → the setting's node and section */
  byKey: Map<string, { node: SettingNode; section: SettingsSection }>
}

/** Inputs are shown unless hidden (`hidden`) or never rendered (`state`). */
function shown(node: Node): node is SettingNode {
  return isInput(node) && node.type !== "state" && !node.hidden
}

export function buildSettingsIndex(schema: SettingsSchema): SettingsIndex {
  const categories: SettingsCategory[] = []
  const sections: SettingsSection[] = []
  const byKey = new Map<string, { node: SettingNode; section: SettingsSection }>()
  const seen = new Set<string>()
  for (const category of schema.categories) {
    const target: SettingsCategory = { id: category.id, label: category.label, sections: [] }
    for (const source of category.sections) {
      // Section ids are unique per WebUI, but guard against an extension reusing one.
      let id = source.id
      for (let n = 2; seen.has(id); n++) id = `${source.id}-${n}`
      seen.add(id)
      const section: SettingsSection = { id, label: source.label, categoryId: category.id, categoryLabel: category.label, items: [] }
      for (const node of source.items as Node[]) {
        if (node.type === "row") {
          const nodes = node.children.filter(shown)
          if (nodes.length === 0) continue
          section.items.push({ kind: "row", keys: nodes.map((n) => n.name), node, nodes })
          for (const child of nodes) byKey.set(child.name, { node: child, section })
        } else if (node.type === "html") {
          section.items.push({ kind: "html", node })
        } else if (node.type === "separator") {
          section.items.push({ kind: "separator", node })
        } else if (shown(node)) {
          section.items.push({ kind: "setting", key: node.name, node })
          byKey.set(node.name, { node, section })
        }
      }
      if (section.items.some((item) => item.kind === "setting" || item.kind === "row")) {
        target.sections.push(section)
        sections.push(section)
      }
    }
    if (target.sections.length > 0) categories.push(target)
  }
  return { categories, sections, byKey }
}

/** Every input of a settings schema, hidden ones included (for the form store). */
export function settingInputs(schema: SettingsSchema): SettingNode[] {
  const out: SettingNode[] = []
  const visit = (node: Node) => {
    if (isInput(node)) out.push(node as SettingNode)
    else if (node.type === "row") node.children.forEach(visit)
  }
  for (const category of schema.categories) for (const section of category.sections) (section.items as Node[]).forEach(visit)
  return out
}

// ---------------------------------------------------------------- search

/** Text of trusted help HTML, for searching (tags and entities dropped). */
export function htmlText(html: string): string {
  return html
    .replace(/<[^>]*>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, " ")
    .trim()
}

/** Lower-cased, whitespace-separated terms of a query (every term must match). */
export function searchTerms(query: string): string[] {
  return query
    .toLowerCase()
    .split(/\s+/)
    .map((term) => term.trim())
    .filter(Boolean)
}

export type Translate = (text: string) => string

/** What a search sees of one setting: its key, label, help, choice labels and descriptions, in English and translated. */
export function settingHaystack(node: SettingNode, translate: Translate): string {
  const parts: string[] = [node.name, node.name.replace(/_/g, " ")]
  const add = (text: string | null | undefined) => {
    if (!text) return
    parts.push(text)
    const translated = translate(text)
    if (translated !== text) parts.push(translated)
  }
  add(node.label)
  add(node.info)
  for (const note of node.notes ?? []) add(note)
  for (const link of node.links ?? []) add(link.label)
  for (const name of node.default_order ?? []) parts.push(name)
  if (node.html) parts.push(htmlText(node.html))
  if ((node.type === "choice" || node.type === "multichoice") && node.choices) {
    for (const choice of node.choices) add(choice.label)
    for (const hint of Object.values(node.choice_hints ?? {})) add(hint)
  }
  return parts.join("\n").toLowerCase()
}

export function matches(haystack: string, terms: readonly string[]): boolean {
  return terms.every((term) => haystack.includes(term))
}

// ---------------------------------------------------------------- filtering

export type FilterMode = "all" | "changed" | "unsaved" | "pinned"

export const FILTER_MODES: readonly FilterMode[] = ["all", "changed", "unsaved", "pinned"]

export type FilterState = {
  query: string
  mode: FilterMode
  /** Values in the editor (unsaved edits included). */
  values: Record<string, unknown>
  /** Values stored on the server. */
  saved: Record<string, unknown>
  /** Built-in defaults (`schema.defaults`). */
  defaults: Record<string, Json> | undefined
  /** Keys pinned to quick settings (mode `pinned`). */
  pinned?: ReadonlySet<string>
  translate: Translate
}

/** Differs from its built-in default (unknown default: never). */
export function isChanged(key: string, values: Record<string, unknown>, defaults: Record<string, Json> | undefined): boolean {
  if (!defaults || !Object.hasOwn(defaults, key)) return false
  return !jsonEqual(values[key], defaults[key])
}

/** Edited but not applied. */
export function isUnsaved(key: string, values: Record<string, unknown>, saved: Record<string, unknown>): boolean {
  return Object.hasOwn(values, key) && !jsonEqual(values[key], saved[key])
}

/** Keys whose editor value differs from the stored one. */
export function unsavedKeys(values: Record<string, unknown>, saved: Record<string, unknown>, known?: ReadonlySet<string>): string[] {
  return Object.keys(values).filter((key) => (!known || known.has(key)) && isUnsaved(key, values, saved))
}

export type FilteredSection = SettingsSection & {
  /** Settings shown (filtered); html/separators only when nothing filters. */
  visible: SettingItem[]
  /** Settings matching, for counts. */
  count: number
}

const haystacks = new WeakMap<SettingNode, { translate: Translate; text: string }>()

function cachedHaystack(node: SettingNode, translate: Translate): string {
  const cached = haystacks.get(node)
  if (cached && cached.translate === translate) return cached.text
  const text = settingHaystack(node, translate)
  haystacks.set(node, { translate, text })
  return text
}

/** The sections and items that pass the search and mode filters. */
export function filterSettings(index: SettingsIndex, filter: FilterState): FilteredSection[] {
  const terms = searchTerms(filter.query)
  const filtering = terms.length > 0 || filter.mode !== "all"
  const keep = (node: SettingNode, section: SettingsSection): boolean => {
    if (filter.mode === "changed" && !isChanged(node.name, filter.values, filter.defaults)) return false
    if (filter.mode === "unsaved" && !isUnsaved(node.name, filter.values, filter.saved)) return false
    if (filter.mode === "pinned" && !filter.pinned?.has(node.name)) return false
    if (terms.length === 0) return true
    // A section or category name matches all its settings ("live previews").
    const context = `${section.label}\n${filter.translate(section.label)}\n${section.categoryLabel}\n${filter.translate(section.categoryLabel)}`.toLowerCase()
    const haystack = cachedHaystack(node, filter.translate)
    return terms.every((term) => haystack.includes(term) || context.includes(term))
  }
  const out: FilteredSection[] = []
  for (const section of index.sections) {
    if (!filtering) {
      const count = section.items.reduce((n, item) => n + (item.kind === "setting" ? 1 : item.kind === "row" ? item.nodes.length : 0), 0)
      out.push({ ...section, visible: section.items, count })
      continue
    }
    const visible: SettingItem[] = []
    let count = 0
    for (const item of section.items) {
      if (item.kind === "setting") {
        if (keep(item.node, section)) {
          visible.push(item)
          count++
        }
      } else if (item.kind === "row") {
        const nodes = item.nodes.filter((node) => keep(node, section))
        if (nodes.length > 0) {
          // A row keeps its layout only while complete; otherwise its matches stand alone.
          if (nodes.length === item.nodes.length) visible.push(item)
          else for (const node of nodes) visible.push({ kind: "setting", key: node.name, node })
          count += nodes.length
        }
      }
    }
    if (visible.length > 0) out.push({ ...section, visible, count })
  }
  return out
}

// ---------------------------------------------------------------- rows

export type SettingsRowItem =
  | { kind: "category"; id: string; label: string }
  | { kind: "section"; id: string; label: string; categoryLabel: string; count: number }
  | { kind: "item"; sectionId: string; item: SettingItem }

/** The flat list the virtualized view renders: category heading, section heading, items. */
export function flattenRows(sections: readonly FilteredSection[]): SettingsRowItem[] {
  const rows: SettingsRowItem[] = []
  let category: string | null = null
  for (const section of sections) {
    if (section.categoryId !== category) {
      category = section.categoryId
      rows.push({ kind: "category", id: section.categoryId, label: section.categoryLabel })
    }
    rows.push({ kind: "section", id: section.id, label: section.label, categoryLabel: section.categoryLabel, count: section.count })
    for (const item of section.visible) rows.push({ kind: "item", sectionId: section.id, item })
  }
  return rows
}

/** Row index of a section's heading (-1 when filtered out). */
export function sectionRowIndex(rows: readonly SettingsRowItem[], sectionId: string): number {
  return rows.findIndex((row) => row.kind === "section" && row.id === sectionId)
}

/** The section a row belongs to (headings belong to the section that follows). */
export function sectionOfRow(rows: readonly SettingsRowItem[], index: number): string | null {
  for (let i = Math.max(0, index); i < rows.length; i++) {
    const row = rows[i]!
    if (row.kind === "section") return row.id
    if (row.kind === "item") return row.sectionId
  }
  return null
}

export function itemKey(item: SettingItem, index: number): string {
  if (item.kind === "setting") return `s:${item.key}`
  if (item.kind === "row") return `r:${item.keys.join(",")}`
  return `${item.kind}:${index}`
}

// ---------------------------------------------------------------- import / export

export const EXPORT_FORMAT = "remi-settings"

export type SettingsExport = {
  format: typeof EXPORT_FORMAT
  version: 1
  exported_at: string
  /** Only settings that differ from their default, unless every value was asked for. */
  values: Record<string, unknown>
}

export function exportSettings(values: Record<string, unknown>, defaults: Record<string, Json> | undefined, { all = false, now = new Date() } = {}): SettingsExport {
  const out: Record<string, unknown> = {}
  for (const key of Object.keys(values).sort()) {
    if (all || !defaults || isChanged(key, values, defaults)) out[key] = values[key]
  }
  return { format: EXPORT_FORMAT, version: 1, exported_at: now.toISOString(), values: out }
}

export type ImportPlan = {
  /** Known settings whose value differs from the current one. */
  changes: Record<string, unknown>
  /** Known settings that already have the imported value. */
  same: string[]
  /** Keys this server does not know (or cannot change). */
  unknown: string[]
}

/**
 * Reads an exported file (ours, or a plain A1111 `config.json` object) and
 * compares it with the current values. `editable` are the keys the editor can
 * change; anything else is reported as unknown.
 */
export function planImport(text: string, current: Record<string, unknown>, editable: ReadonlySet<string>): ImportPlan {
  let data: unknown
  try {
    data = JSON.parse(text)
  } catch {
    throw new Error("The file is not JSON.")
  }
  if (!data || typeof data !== "object" || Array.isArray(data)) throw new Error("The file does not contain settings.")
  const record = data as Record<string, unknown>
  const values = record.format === EXPORT_FORMAT && record.values && typeof record.values === "object" ? (record.values as Record<string, unknown>) : record
  const plan: ImportPlan = { changes: {}, same: [], unknown: [] }
  for (const [key, value] of Object.entries(values)) {
    if (!editable.has(key)) plan.unknown.push(key)
    else if (jsonEqual(value, current[key])) plan.same.push(key)
    else plan.changes[key] = value
  }
  return plan
}

