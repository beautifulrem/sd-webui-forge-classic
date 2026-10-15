import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { createFormStore, type FormStore } from "@/ui/form-store"
import { TabForm } from "@/ui/SchemaForm"
import type { Json, TabSchema } from "@/ui/schema"
import { stageWindow } from "../_anima/schedule"
import { widgetKeys } from "../_lib/i18n"
import zh from "./locales/zh-CN.json"
import source from "./index.tsx?raw"
import { artistNames, type Choices, rowOf, windowOf, withAuto, withStage } from "./rows"

const OWNER = "anima_artist_scheduled_mixer"
// The props the Mixer's params() gives the base table
const props = {
  chain: "base",
  shift: ["core.distilled_cfg_scale"],
  steps: ["core.steps"],
  max_rows: 32,
  optimization_param: "optimization",
  default_blocks: { Performance: "10-18", Balance: "0-51", Quality: "0-51" },
  max_artists: { Performance: 6, Balance: 10, Quality: 16 },
  curves: [
    { value: "Smooth", label: "Smooth", shape: "smooth" },
    { value: "Hold", label: "Hold", shape: "hold" },
    { value: "Triangle", label: "Triangle", shape: "triangle" },
    { value: "Front loaded", label: "Front loaded", shape: "front" },
    { value: "Back loaded", label: "Back loaded", shape: "back" },
  ],
  curve_floor: 0.25,
  stages: [
    { value: "Custom", label: "Custom" },
    { value: "Composition", label: "Composition", kind: "composition" as const },
    { value: "Character", label: "Character", kind: "character" as const },
    { value: "Style", label: "Style", kind: "style" as const },
  ],
}
const choices: Choices = { curves: props.curves, stages: props.stages }
const row = (artist: string, extra: Record<string, Json> = {}): Record<string, Json> => ({ enabled: true, artist, weight: 1, blocks: "0-51", start: 0, end: 1, peak: 0.5, curve: "Hold", stage: "Custom", auto_shift: false, ...extra })

function renderTable(rows: Json[], optimization = "Balance"): FormStore {
  const schema: TabSchema = {
    version: 1,
    tab: "txt2img",
    core: {
      owner: "core",
      layout: { type: "column", children: [{ type: "slider", name: "distilled_cfg_scale", label: "Distilled CFG Scale", minimum: 1, maximum: 24, step: 0.5, integer: false }, { type: "slot", section: "scripts" }] },
      defaults: { distilled_cfg_scale: 3, steps: 20 },
    },
    scripts: [
      {
        id: OWNER,
        title: "Anima Artist Scheduled Mixer",
        kind: "alwayson",
        section: null,
        order: 0,
        legacy: false,
        layout: {
          type: "section",
          title: "Anima Artist Scheduled Mixer",
          open: true,
          children: [
            { type: "choice", name: "optimization", label: "Optimization preset", choices: Object.keys(props.default_blocks).map((value) => ({ value, label: value })), style: "dropdown", allow_custom: false },
            { type: "custom", name: "base_rows", label: "Base artists", widget: "artist-mix-table", props: props as unknown as Record<string, Json> },
          ],
        },
        defaults: { optimization, base_rows: rows },
        args: ["optimization", "base_rows"],
      },
    ],
  }
  const store = createFormStore(`artist-mix-table-${Math.random()}`, { persistDrafts: false })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, enabled: false } } })
  render(
    <QueryClientProvider client={client}>
      <TabForm schema={schema} formId="txt2img" store={store} />
    </QueryClientProvider>,
  )
  return store
}

const rows = (store: FormStore) => store.getState().values[OWNER]?.base_rows as Record<string, unknown>[]

async function table() {
  return within(await screen.findByRole("group", { name: "Base artists" }, { timeout: 40_000 }))
}

describe("rows", () => {
  it("read the Gradio panel's labels and Yes/No values", () => {
    expect(rowOf({ enabled: "No", artist: "x", weight: "9", curve: "triangle", stage: "style", auto_shift: "Yes" }, "0-51", choices)).toEqual({
      enabled: false,
      artist: "x",
      weight: 4,
      blocks: "0-51",
      start: 0,
      end: 1,
      peak: 0.5,
      curve: "Triangle",
      stage: "Style",
      auto_shift: true,
    })
  })

  it("take a stage's window and freeze it when Auto Shift goes off", () => {
    const styled = withStage(rowOf(row("a"), "0-51", choices), "Style", 5, choices)
    const style = stageWindow("style", 5)
    expect(styled).toMatchObject({ stage: "Style", auto_shift: true, start: Number(style.start.toFixed(4)), end: 1 })
    expect(windowOf(styled, 9, choices).start).toBe(Number(stageWindow("style", 9).start.toFixed(4)))
    expect(withAuto(styled, false, 9, choices)).toMatchObject({ auto_shift: false, start: Number(stageWindow("style", 9).start.toFixed(4)) })
    expect(withStage(styled, "Custom", 5, choices).auto_shift).toBe(false)
  })

  it("count the artists of comma-separated chains", () => {
    expect(artistNames([rowOf(row("a, b，c"), "", choices), rowOf(row("d", { enabled: false }), "", choices)])).toEqual(["a", "b", "c"])
  })
})

describe("artist-mix-table", { timeout: 60_000 }, () => {
  // jsdom has no layout: give the scroller a size so the virtualizer renders rows
  beforeEach(() => {
    vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockReturnValue(600)
    vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockReturnValue(800)
  })
  afterEach(() => vi.restoreAllMocks())

  it("adds, edits, reorders and removes rows", async () => {
    const store = renderTable([row("wlop"), row("")])
    const group = await table()
    fireEvent.change(group.getByRole("textbox", { name: "Row 2 artist" }), { target: { value: "(sakimichan:0.8)" } })
    expect(rows(store)[1]).toMatchObject({ artist: "(sakimichan:0.8)" })
    fireEvent.click(group.getByRole("button", { name: "Move row 2 up" }))
    expect(rows(store).map((r) => r.artist)).toEqual(["(sakimichan:0.8)", "wlop"])
    fireEvent.click(group.getByRole("button", { name: "Add row" }))
    expect(rows(store)).toHaveLength(3)
    expect(rows(store)[2]).toMatchObject({ artist: "", blocks: "0-51", stage: "Custom", auto_shift: false })
    fireEvent.click(group.getByRole("button", { name: "Duplicate row 1" }))
    expect(rows(store).map((r) => r.artist)).toEqual(["(sakimichan:0.8)", "(sakimichan:0.8)", "wlop", ""])
    fireEvent.click(group.getByRole("button", { name: "Remove row 4" }))
    fireEvent.click(group.getByRole("switch", { name: "Row 3 enabled" }))
    expect(rows(store)).toHaveLength(3)
    expect(rows(store)[2]).toMatchObject({ artist: "wlop", enabled: false })
    expect(group.getByRole("status")).toHaveTextContent("2 artists mix")
  })

  it("times a stage row from the generation form's Shift", async () => {
    const store = renderTable([row("wlop")])
    const group = await table()
    fireEvent.change(group.getByRole("combobox", { name: "Row 1 stage" }), { target: { value: "Composition" } })
    const composition = stageWindow("composition", 3)
    expect(rows(store)[0]).toMatchObject({ stage: "Composition", auto_shift: true, start: 0, end: Number(composition.end.toFixed(4)) })
    expect(group.getByRole("textbox", { name: "Row 1 end" })).toBeDisabled()

    act(() => store.getState().setValue("core", "distilled_cfg_scale", 12))
    expect(group.getByRole("textbox", { name: "Row 1 end" })).toHaveValue(String(Number(stageWindow("composition", 12).end.toFixed(4))))
    fireEvent.click(group.getByRole("switch", { name: "Row 1 auto shift" }))
    expect(rows(store)[0]).toMatchObject({ auto_shift: false, end: Number(stageWindow("composition", 12).end.toFixed(4)) })
    const end = group.getByRole("textbox", { name: "Row 1 end" })
    fireEvent.change(end, { target: { value: "0.5" } })
    fireEvent.blur(end)
    expect(rows(store)[0]).toMatchObject({ end: 0.5 })
  })

  it("warns when the optimization preset caps the artists and fills new rows with its Blocks", async () => {
    const store = renderTable(Array.from({ length: 7 }, (_, i) => row(`artist ${i}`)), "Performance")
    const group = await table()
    expect(group.getByRole("status")).toHaveTextContent("7 artists: the preset mixes the first 6")
    fireEvent.click(group.getByRole("button", { name: "Add row" }))
    expect(rows(store)[7]).toMatchObject({ blocks: "10-18" })
  })

  it("stops at the row limit", async () => {
    renderTable(Array.from({ length: 32 }, (_, i) => row(`a${i}`)))
    const group = await table()
    expect(group.getByRole("button", { name: "Add row" })).toBeDisabled()
    expect(group.getByText("32 of 32 rows")).toBeInTheDocument()
  })

  it("translates its chrome", () => {
    const keys = widgetKeys(source)
    expect(keys.length).toBeGreaterThan(20)
    expect(keys.filter((key) => !(key in zh))).toEqual([])
  })
})
