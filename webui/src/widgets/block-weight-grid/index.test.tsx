import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, within } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { createFormStore, type FormStore } from "@/ui/form-store"
import { TabForm } from "@/ui/SchemaForm"
import type { Node, TabSchema } from "@/ui/schema"
import { widgetKeys } from "../_lib/i18n"
import zh from "./locales/zh-CN.json"
import source from "./index.tsx?raw"
import chartSource from "./WeightChart.tsx?raw"

const MAPPING_52 = [0, 1, 1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 7, 7, 8, 9, 9, 9, 10, 11, 11, 11, 12, 13, 14, 14, 14, 15, 16, 16, 16, 17, 18, 18, 18, 19, 20, 20, 20, 21, 22, 22, 22, 23, 24, 24, 24, 25, 26, 27]

// The props anima-lora-layer-weight's params() gives its widgets
const blocks: Node = {
  type: "custom",
  name: "lora_block_weights",
  label: "Anima block weights",
  widget: "block-weight-grid",
  props: {
    mode: "blocks",
    layouts: [28, 40, 52],
    mappings: { "52": MAPPING_52 },
    bands: [
      { label: "Composition", start: 0, end: 9 },
      { label: "Character", start: 10, end: 18 },
      { label: "Style", start: 19, end: 27 },
    ],
    presets: [
      { label: "Flat 1.0", value: "all=1.0" },
      { label: "Balanced", value: "0-9=0.6,10-18=1.0,19-27=0.75" },
    ],
  },
}
const modules: Node = { type: "custom", name: "lora_module_weights", label: "Module weights", widget: "block-weight-grid", props: { mode: "modules", modules: ["self_attn", "cross_attn", "mlp", "adaln", "norm"] } }
const qwen: Node = { type: "custom", name: "lora_text_layer_weights", label: "Qwen text-encoder layer weights", widget: "block-weight-grid", props: { mode: "layers", count: 28 } }

function renderWidgets(defaults: Record<string, string>): FormStore {
  const schema: TabSchema = {
    version: 1,
    tab: "txt2img",
    core: { owner: "core", layout: { type: "column", children: [{ type: "slot", section: "scripts" }] }, defaults: {} },
    scripts: [
      {
        id: "anima_lora_layer_weight",
        title: "Anima LoRA Layer Weight",
        kind: "alwayson",
        section: null,
        order: 0,
        layout: { type: "section", title: "Anima LoRA/LoKR Layer Weight", open: true, children: [blocks, modules, qwen] },
        defaults,
        args: Object.keys(defaults),
      },
    ],
  }
  const store = createFormStore(`block-weight-grid-${Math.random()}`, { persistDrafts: false })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, enabled: false } } })
  render(
    <QueryClientProvider client={client}>
      <TabForm schema={schema} formId="txt2img" store={store} />
    </QueryClientProvider>,
  )
  return store
}

const DEFAULTS = { lora_block_weights: "all=1.0", lora_module_weights: "self_attn=1.0\ncross_attn=1.0\nmlp=1.0\nadaln=1.0", lora_text_layer_weights: "" }
const value = (store: FormStore, name: string) => store.getState().values.anima_lora_layer_weight?.[name]

async function blockGroup() {
  return within(await screen.findByRole("group", { name: "Anima block weights" }, { timeout: 40_000 }))
}

// The first test also loads the widget chunk lazily
describe("block-weight-grid", { timeout: 60_000 }, () => {
  it("edits one block from the keyboard and writes the canonical text", async () => {
    const store = renderWidgets(DEFAULTS)
    const group = await blockGroup()
    const bars = group.getAllByRole("slider")
    expect(bars).toHaveLength(28)
    expect(bars[3]).toHaveAttribute("aria-valuenow", "1")
    act(() => bars[3]!.focus())
    fireEvent.keyDown(bars[3]!, { key: "ArrowDown" })
    expect(value(store, "lora_block_weights")).toBe("all=1.0, 3=0.95")
    fireEvent.keyDown(bars[3]!, { key: "ArrowRight" })
    expect(document.activeElement).toBe(bars[4])
    fireEvent.keyDown(bars[4]!, { key: "Home" })
    expect(value(store, "lora_block_weights")).toBe("all=1.0, 3=0.95, 4=0.0")
  })

  it("sets a depth band at once and applies presets", async () => {
    const store = renderWidgets(DEFAULTS)
    const group = await blockGroup()
    const composition = group.getByRole("textbox", { name: "Composition (0-9)" })
    fireEvent.change(composition, { target: { value: "0.6" } })
    fireEvent.blur(composition)
    expect(value(store, "lora_block_weights")).toBe("all=1.0, 0-9=0.6")
    fireEvent.click(group.getByRole("button", { name: "Balanced" }))
    expect(value(store, "lora_block_weights")).toBe("0-9=0.6,10-18=1.0,19-27=0.75")
  })

  it("paints a stroke across the bars", async () => {
    const store = renderWidgets(DEFAULTS)
    const group = await blockGroup()
    const chart = group.getByTestId("weight-chart")
    chart.getBoundingClientRect = () => ({ left: 0, top: 0, width: 280, height: 100, right: 280, bottom: 100, x: 0, y: 0, toJSON: () => ({}) })
    // 10 px per block; y 75 of 100 is a quarter of the 0-2 range: 0.5
    fireEvent.pointerDown(chart, { button: 0, buttons: 1, clientX: 5, clientY: 75, pointerId: 1 })
    fireEvent.pointerMove(chart, { buttons: 1, clientX: 45, clientY: 75, pointerId: 1 })
    expect(value(store, "lora_block_weights")).toBe("all=1.0")  // committed on release
    fireEvent.pointerUp(chart, { pointerId: 1 })
    expect(value(store, "lora_block_weights")).toBe("all=1.0, 0-4=0.5")
  })

  it("shows 40/52-block LoRAs with their inserted copies", async () => {
    renderWidgets({ ...DEFAULTS, lora_block_weights: "all=1.0, 45=0.5" })
    const group = await blockGroup()
    const bars = group.getAllByRole("slider")
    expect(bars).toHaveLength(52)  // index 45 needs the 52-block layout
    expect(bars[45]).toHaveAttribute("aria-valuenow", "0.5")
    expect(bars[2]!.getAttribute("title")).toContain("↳ 1")
    fireEvent.click(group.getByRole("radio", { name: "28 blocks" }))
    expect(group.getAllByRole("slider")).toHaveLength(28)
  })

  it("keeps the blocks past the chosen layout when editing", async () => {
    const store = renderWidgets({ ...DEFAULTS, lora_block_weights: "all=1.0, 45=0.5" })
    const group = await blockGroup()
    fireEvent.click(group.getByRole("radio", { name: "28 blocks" }))
    expect(group.getByRole("note")).toHaveTextContent("45=0.5")
    const bars = group.getAllByRole("slider")
    act(() => bars[0]!.focus())
    fireEvent.keyDown(bars[0]!, { key: "ArrowDown" })
    expect(value(store, "lora_block_weights")).toBe("all=1.0, 0=0.95, 45=0.5")
  })

  it("never clamps a weight outside 0-2 it shows", async () => {
    const store = renderWidgets({ ...DEFAULTS, lora_block_weights: "all=1.0, 0=-0.5, 1=3.0" })
    const group = await blockGroup()
    const bars = group.getAllByRole("slider")
    expect(bars[0]).toHaveAttribute("aria-valuemin", "-0.5")
    expect(bars[1]).toHaveAttribute("aria-valuemax", "3")
    act(() => bars[0]!.focus())
    fireEvent.keyDown(bars[0]!, { key: "ArrowUp" })
    expect(value(store, "lora_block_weights")).toBe("all=1.0, 0=-0.45, 1=3.0")
    fireEvent.keyDown(bars[0]!, { key: "ArrowRight" })
    fireEvent.keyDown(bars[1]!, { key: "ArrowDown" })
    expect(value(store, "lora_block_weights")).toBe("all=1.0, 0=-0.45, 1=2.95")
  })

  it("edits the text itself and flags what the grid cannot show", async () => {
    const store = renderWidgets({ ...DEFAULTS, lora_block_weights: "0.5, 3=0.2" })
    const group = await blockGroup()
    expect(group.getByRole("note")).toHaveTextContent("0.5")
    fireEvent.click(group.getByRole("radio", { name: "Text" }))
    const text = group.getByRole("textbox")
    expect(text).toHaveValue("0.5, 3=0.2")
    fireEvent.change(text, { target: { value: "all=0.8" } })
    expect(value(store, "lora_block_weights")).toBe("all=0.8")
  })

  it("shows norm and keeps the modules it has no row for", async () => {
    const store = renderWidgets({ ...DEFAULTS, lora_module_weights: "norm=0.5\nfinal_layer=0.3\nmlp=oops" })
    const moduleGroup = within(await screen.findByRole("group", { name: "Module weights" }, { timeout: 40_000 }))
    expect(moduleGroup.getByRole("textbox", { name: "norm" })).toHaveValue("0.5")
    const [kept, ignored] = moduleGroup.getAllByRole("note")
    expect(kept).toHaveTextContent("a run applies them; editing the grid keeps them): final_layer=0.3")
    expect(ignored).toHaveTextContent("a run ignores them; editing the grid drops them): mlp=oops")
    const mlp = moduleGroup.getByRole("textbox", { name: "mlp" })
    fireEvent.change(mlp, { target: { value: "0.7" } })
    fireEvent.blur(mlp)
    expect(value(store, "lora_module_weights")).toBe("mlp=0.7\nnorm=0.5\nfinal_layer=0.3")
  })

  it("edits module weights and Qwen layers", async () => {
    const store = renderWidgets(DEFAULTS)
    const moduleGroup = within(await screen.findByRole("group", { name: "Module weights" }, { timeout: 40_000 }))
    const mlp = moduleGroup.getByRole("textbox", { name: "mlp" })
    fireEvent.change(mlp, { target: { value: "0.7" } })
    fireEvent.blur(mlp)
    expect(value(store, "lora_module_weights")).toBe("self_attn=1.0\ncross_attn=1.0\nmlp=0.7\nadaln=1.0")

    const layers = within(await screen.findByRole("group", { name: "Qwen text-encoder layer weights" }))
    const bars = layers.getAllByRole("slider", { name: /^Layer / })
    expect(bars).toHaveLength(28)
    fireEvent.keyDown(bars[0]!, { key: "PageDown" })
    expect(value(store, "lora_text_layer_weights")).toBe("all=1.0, 0=0.75")
    fireEvent.keyDown(bars[0]!, { key: "PageUp" })
    expect(value(store, "lora_text_layer_weights")).toBe("")  // flat again: the text encoder is left unchanged
  })

  it("translates its chrome", () => {
    const keys = widgetKeys(source)
    expect(keys.length).toBeGreaterThan(8)
    expect(keys.filter((key) => !(key in zh))).toEqual([])
    expect(widgetKeys(chartSource)).toEqual([])
  })
})
