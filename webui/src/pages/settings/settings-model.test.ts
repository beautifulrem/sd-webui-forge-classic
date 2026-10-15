import { describe, expect, it } from "vitest"
import type { SettingsSchema } from "@/ui/schema"
import {
  buildSettingsIndex,
  exportSettings,
  filterSettings,
  flattenRows,
  htmlText,
  isChanged,
  planImport,
  translateHtml,
  searchTerms,
  sectionOfRow,
  sectionRowIndex,
  settingHaystack,
  settingInputs,
  unsavedKeys,
  type FilterState,
} from "./settings-model"

const schema: SettingsSchema = {
  version: 1,
  categories: [
    {
      id: "sd",
      label: "Stable Diffusion",
      sections: [
        {
          id: "saving-images",
          label: "Saving images/grids",
          items: [
            { type: "slider", name: "jpeg_quality", label: "Quality for saved jpeg images", default: 80, minimum: 1, maximum: 100, step: 1, integer: true, notes: ["also used for webp"] },
            { type: "toggle", name: "samples_save", label: "Always save all generated images", default: true },
            { type: "html", text: "<b>Grids</b>" },
            {
              type: "row",
              children: [
                { type: "number", name: "grid_rows", label: "Grid row count", default: -1 },
                { type: "text", name: "grid_text", label: "Grid text", default: "", hidden: true },
              ],
            },
            { type: "choice", name: "samples_format", label: "File format", default: "png", choices: [{ value: "png", label: "PNG" }, { value: "webp", label: "WebP image" }], html: "<span class='info'>(see <b>Saving</b> &amp; more)</span>" },
            { type: "state", name: "internal", label: "", default: null },
          ],
        },
      ],
    },
    {
      id: "ui",
      label: "User Interface",
      sections: [
        { id: "ui", label: "Live previews", items: [{ type: "toggle", name: "live_previews_enable", label: "Show live previews", default: true, needs_reload_ui: true }] },
        { id: "empty", label: "Only help", items: [{ type: "html", text: "nothing to set" }] },
      ],
    },
  ],
  quicksettings: [],
  values: { jpeg_quality: 80, samples_save: true, grid_rows: -1, grid_text: "", samples_format: "png", live_previews_enable: true },
  defaults: { jpeg_quality: 80, samples_save: true, grid_rows: -1, grid_text: "", samples_format: "png", live_previews_enable: true },
}

const zh: Record<string, string> = { "Quality for saved jpeg images": "保存的 JPEG 图像质量", "Live previews": "实时预览" }
const translate = (text: string) => zh[text] ?? text

function filter(overrides: Partial<FilterState> = {}): FilterState {
  return { query: "", mode: "all", values: { ...schema.values }, saved: { ...schema.values }, defaults: schema.defaults, translate, ...overrides }
}

const keysOf = (sections: ReturnType<typeof filterSettings>) =>
  sections.flatMap((section) => section.visible.flatMap((item) => (item.kind === "setting" ? [item.key] : item.kind === "row" ? item.keys : [])))

describe("settings index", () => {
  const index = buildSettingsIndex(schema)

  it("keeps sections with settings and skips hidden and state inputs", () => {
    expect(index.sections.map((s) => s.id)).toEqual(["saving-images", "ui"])
    expect([...index.byKey.keys()]).toEqual(["jpeg_quality", "samples_save", "grid_rows", "samples_format", "live_previews_enable"])
    const row = index.sections[0]!.items.find((item) => item.kind === "row")
    expect(row?.kind === "row" && row.keys).toEqual(["grid_rows"])
  })

  it("lists every input for the form store, hidden ones included", () => {
    expect(settingInputs(schema).map((node) => node.name)).toEqual(["jpeg_quality", "samples_save", "grid_rows", "grid_text", "samples_format", "internal", "live_previews_enable"])
  })

  it("disambiguates a reused section id", () => {
    const twice = buildSettingsIndex({ ...schema, categories: [schema.categories[1]!, { ...schema.categories[1]!, id: "x" }] })
    expect(twice.sections.map((s) => s.id)).toEqual(["ui", "ui-2"])
  })
})

describe("settings search", () => {
  const index = buildSettingsIndex(schema)

  it("splits terms and requires all of them", () => {
    expect(searchTerms("  JPEG   quality ")).toEqual(["jpeg", "quality"])
    expect(keysOf(filterSettings(index, filter({ query: "jpeg quality" })))).toEqual(["jpeg_quality"])
    expect(keysOf(filterSettings(index, filter({ query: "jpeg nonsense" })))).toEqual([])
  })

  it("matches keys, notes, choice labels, help HTML text and translations", () => {
    expect(keysOf(filterSettings(index, filter({ query: "samples_save" })))).toEqual(["samples_save"])
    expect(keysOf(filterSettings(index, filter({ query: "samples save" })))).toEqual(["samples_save"])
    expect(keysOf(filterSettings(index, filter({ query: "webp" })))).toEqual(["jpeg_quality", "samples_format"])
    expect(keysOf(filterSettings(index, filter({ query: "saving & more" })))).toEqual(["samples_format"])
    expect(keysOf(filterSettings(index, filter({ query: "图像质量" })))).toEqual(["jpeg_quality"])
  })

  it("matches every setting of a section whose name matches", () => {
    expect(keysOf(filterSettings(index, filter({ query: "实时预览" })))).toEqual(["live_previews_enable"])
    expect(keysOf(filterSettings(index, filter({ query: "user interface" })))).toEqual(["live_previews_enable"])
  })

  it("drops help-only items while filtering and keeps them otherwise", () => {
    const all = filterSettings(index, filter())
    expect(all[0]!.visible.some((item) => item.kind === "html")).toBe(true)
    expect(all[0]!.count).toBe(4)
    const found = filterSettings(index, filter({ query: "row count" }))
    expect(found[0]!.visible.map((item) => item.kind)).toEqual(["row"])
  })

  it("builds a haystack without markup", () => {
    expect(htmlText("<span>(a &amp; <b>b</b>)</span>")).toBe("(a & b )")
    expect(settingHaystack(index.byKey.get("jpeg_quality")!.node, translate)).toContain("jpeg quality")
  })
})

describe("filter modes", () => {
  const index = buildSettingsIndex(schema)

  it("changed = differs from the built-in default", () => {
    const values = { ...schema.values, jpeg_quality: 95 }
    expect(isChanged("jpeg_quality", values, schema.defaults)).toBe(true)
    expect(isChanged("jpeg_quality", values, undefined)).toBe(false)
    expect(keysOf(filterSettings(index, filter({ mode: "changed", values, saved: values })))).toEqual(["jpeg_quality"])
  })

  it("unsaved = editor value differs from the stored one", () => {
    const values = { ...schema.values, samples_save: false, grid_rows: 2 }
    expect(unsavedKeys(values, schema.values).sort()).toEqual(["grid_rows", "samples_save"])
    expect(unsavedKeys(values, schema.values, new Set(["grid_rows"]))).toEqual(["grid_rows"])
    expect(keysOf(filterSettings(index, filter({ mode: "unsaved", values })))).toEqual(["samples_save", "grid_rows"])
  })

  it("pinned = in quick settings", () => {
    expect(keysOf(filterSettings(index, filter({ mode: "pinned", pinned: new Set(["live_previews_enable"]) })))).toEqual(["live_previews_enable"])
  })

  it("a partly matching row splits into single settings", () => {
    const rowSchema: SettingsSchema = {
      ...schema,
      categories: [{ id: "c", label: "C", sections: [{ id: "s", label: "S", items: [{ type: "row", children: [{ type: "toggle", name: "a", label: "Alpha", default: false }, { type: "toggle", name: "b", label: "Beta", default: false }] }] }] }],
    }
    const sections = filterSettings(buildSettingsIndex(rowSchema), filter({ query: "beta", values: { a: false, b: false }, saved: { a: false, b: false } }))
    expect(sections[0]!.visible).toEqual([expect.objectContaining({ kind: "setting", key: "b" })])
  })
})

describe("rows for the virtualized list", () => {
  it("puts category and section headings before the items and maps rows back to sections", () => {
    const rows = flattenRows(filterSettings(buildSettingsIndex(schema), filter()))
    expect(rows.map((row) => row.kind)).toEqual(["category", "section", "item", "item", "item", "item", "item", "category", "section", "item"])
    expect(sectionRowIndex(rows, "ui")).toBe(8)
    expect(sectionRowIndex(rows, "nope")).toBe(-1)
    expect(sectionOfRow(rows, 0)).toBe("saving-images")
    expect(sectionOfRow(rows, 7)).toBe("ui")
    expect(sectionOfRow(rows, 9)).toBe("ui")
  })
})

describe("import and export", () => {
  it("exports changed values only unless asked for all", () => {
    const values = { ...schema.values, jpeg_quality: 95 }
    const now = new Date("2026-10-15T00:00:00Z")
    expect(exportSettings(values, schema.defaults, { now })).toEqual({ format: "remi-settings", version: 1, exported_at: "2026-10-15T00:00:00.000Z", values: { jpeg_quality: 95 } })
    expect(Object.keys(exportSettings(values, schema.defaults, { all: true, now }).values)).toHaveLength(6)
  })

  it("plans an import of our files and of plain config.json objects", () => {
    const editable = new Set(["jpeg_quality", "samples_save"])
    const ours = JSON.stringify({ format: "remi-settings", version: 1, values: { jpeg_quality: 50, samples_save: true, gone: 1 } })
    expect(planImport(ours, schema.values, editable)).toEqual({ changes: { jpeg_quality: 50 }, same: ["samples_save"], unknown: ["gone"] })
    expect(planImport('{"samples_save": false}', schema.values, editable).changes).toEqual({ samples_save: false })
    expect(() => planImport("nope", schema.values, editable)).toThrow("The file is not JSON.")
    expect(() => planImport("[1]", schema.values, editable)).toThrow("The file does not contain settings.")
  })
})

describe("translateHtml", () => {
  const catalog: Record<string, string> = { "<b>e.g.</b> for <b>X/Y/Z Plot</b>": "<b>例如</b> 用于 <b>X/Y/Z Plot</b>", "<i>whole</i>": "<i>整体</i>" }
  const t = (key: string) => catalog[key] ?? key

  it("translates the text inside a legacy info span, or the whole comment", () => {
    expect(translateHtml("<span class='info'>(<b>e.g.</b> for <b>X/Y/Z Plot</b>)</span>", t)).toBe("<span class='info'>(<b>例如</b> 用于 <b>X/Y/Z Plot</b>)</span>")
    expect(translateHtml("<i>whole</i>", t)).toBe("<i>整体</i>")
    expect(translateHtml("<span class='info'>(untranslated)</span>", t)).toBe("<span class='info'>(untranslated)</span>")
  })
})
