import { act, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { FormProvider, OwnerProvider } from "@/ui/context"
import { createFormStore, type FormStore } from "@/ui/form-store"
import type { CustomNode, Node } from "@/ui/schema"
import { widgetKeys } from "../_lib/i18n"
import zh from "./locales/zh-CN.json"
import ResolutionPresets from "./index"
import { aspect, presetOptions, type Size } from "./sizes"

const OWNER = "anima_random_resolution"
const STANDARD: Size[] = [[1088, 1088], [1216, 832], [832, 1216]]
const HIGHRES: Size[] = [[1536, 1536], [1856, 1216]]
const PROPS = { standard: STANDARD, highres: HIGHRES, width: "core.width", height: "core.height" }
const NODE: CustomNode = { type: "custom", name: "presets", label: "", widget: "resolution-presets", props: PROPS, default: null }

function mount(size = { width: 1024, height: 1024 }): FormStore {
  const store = createFormStore(`presets-${Math.random()}`, { persistDrafts: false })
  const core: Node = {
    type: "group",
    children: [
      { type: "slider", name: "width", label: "Width", minimum: 64, maximum: 2048, step: 64, integer: true, on_change: "hr_resolution" },
      { type: "slider", name: "height", label: "Height", minimum: 64, maximum: 2048, step: 64, integer: true, on_change: "hr_resolution" },
    ],
  }
  store.getState().load([
    { owner: "core", layout: core, defaults: size },
    { owner: OWNER, layout: { type: "group", children: [NODE] }, defaults: { presets: null } },
  ])
  render(
    <FormProvider formId="txt2img" store={store}>
      <OwnerProvider owner={OWNER}>
        <ResolutionPresets value={null} onChange={() => {}} props={PROPS} node={NODE} disabled={false} id="presets" label="" />
      </OwnerProvider>
    </FormProvider>,
  )
  return store
}

afterEach(() => vi.restoreAllMocks())

// The first render (jsdom, Radix) can take seconds on a loaded machine
describe("resolution-presets", { timeout: 20_000 }, () => {
  it("formats the presets", () => {
    expect(aspect(1216, 832)).toBe("19:13")
    expect(presetOptions([[1088, 1088], [0, 5]])).toEqual([{ value: "1088x1088", label: "1088 × 1088", hint: "1.18 MP · 1:1" }])
  })

  it("writes width and height at once and runs their change effect once", async () => {
    const user = userEvent.setup()
    const action = vi.spyOn(api, "action").mockResolvedValue({})
    const store = mount()
    await user.click(screen.getByRole("combobox", { name: "Preset" }))
    await user.click(await screen.findByRole("option", { name: /1216 × 832/ }))
    expect(store.getState().values.core).toMatchObject({ width: 1216, height: 832 })
    await waitFor(() => expect(action).toHaveBeenCalledTimes(1))
    expect(action.mock.calls[0]!.slice(0, 2)).toEqual(["core", "hr_resolution"])
  })

  it("shows the preset matching the current size, in its list only", () => {
    const store = mount({ width: 1536, height: 1536 })
    expect(screen.getByRole("combobox", { name: /High-res preset/ })).toHaveTextContent("1536 × 1536")
    expect(screen.getByRole("combobox", { name: "Preset" })).toHaveTextContent("Choose a size")
    act(() => store.getState().setValues("core", { width: 832, height: 1216 }))
    expect(screen.getByRole("combobox", { name: "Preset" })).toHaveTextContent("832 × 1216")
  })

  it("has a zh-CN translation for every chrome string", () => {
    const sources = import.meta.glob<string>(["./*.tsx", "./*.ts", "!./*.test.*"], { query: "?raw", import: "default", eager: true })
    const keys = Object.values(sources).flatMap(widgetKeys)
    expect(keys.length).toBeGreaterThan(0)
    expect(keys.filter((key) => !(zh as Record<string, string>)[key])).toEqual([])
  })
})
