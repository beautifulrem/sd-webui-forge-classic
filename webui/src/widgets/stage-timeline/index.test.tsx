import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { createFormStore, type FormStore } from "@/ui/form-store"
import { TabForm } from "@/ui/SchemaForm"
import type { TabSchema } from "@/ui/schema"
import { stageWindow } from "../_anima/schedule"
import { widgetKeys } from "../_lib/i18n"
import zh from "./locales/zh-CN.json"
import source from "./index.tsx?raw"
import editorSource from "./TemplateEditor.tsx?raw"
import fixture from "./model.fixture.json"

const STYLE_ONLY = "仅画风阶段介入"
const SMOOTH = "平滑淡入淡出"
const OWNER = "anima_lora_stage_scheduler"

// The props anima-lora-stage-scheduler's params() gives the base panel
const props = {
  panel: "base",
  template_param: "base_template",
  shift: ["core.distilled_cfg_scale"],
  steps: ["core.steps"],
  strength_max: 2,
  curves: [
    { value: SMOOTH, label: "Smooth fade in/out", shape: "smooth" },
    { value: "保持", label: "Hold", shape: "hold" },
  ],
  stages: [
    { value: "构图阶段", label: "Composition stage", kind: "composition" },
    { value: "人物特征阶段", label: "Character stage", kind: "character" },
    { value: "画风阶段", label: "Style stage", kind: "style" },
  ],
  builder: [
    { value: STYLE_ONLY, label: "Style stage only", stage: "画风阶段", mode: "allow" },
    { value: "仅禁止画风阶段介入", label: "Block the style stage", stage: "画风阶段", mode: "ban" },
    { value: "仅人物特征阶段介入", label: "Character stage only", stage: "人物特征阶段", mode: "allow" },
  ],
  templates: fixture.templates,
}

const schema: TabSchema = {
  version: 1,
  tab: "txt2img",
  core: {
    owner: "core",
    layout: {
      type: "column",
      children: [
        { type: "slider", name: "distilled_cfg_scale", label: "Distilled CFG Scale", minimum: 1, maximum: 24, step: 0.5, integer: false },
        { type: "slot", section: "scripts" },
      ],
    },
    defaults: { distilled_cfg_scale: 3, steps: 21 },
  },
  scripts: [
    {
      id: OWNER,
      title: "Anima LoRA Stage Scheduler",
      kind: "alwayson",
      section: null,
      order: 0,
      legacy: false,
      layout: {
        type: "section",
        title: "Anima LoRA Stage Scheduler",
        open: true,
        children: [
          { type: "choice", name: "base_template", label: "Template", choices: Object.keys(fixture.templates).map((value) => ({ value, label: value })), style: "dropdown", allow_custom: false },
          { type: "custom", name: "base_timing", label: "Stage timing", widget: "stage-timeline", props },
        ],
      },
      defaults: { base_template: STYLE_ONLY, base_timing: { auto: true, start: 0.62, end: 1, peak: 0.86, strength: 1, curve: SMOOTH } },
      args: ["base_template", "base_timing"],
    },
  ],
}

function renderPanel(values: Record<string, unknown> = {}): FormStore {
  const store = createFormStore(`stage-timeline-${Math.random()}`, { persistDrafts: false })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, enabled: false } } })
  render(
    <QueryClientProvider client={client}>
      <TabForm schema={schema} formId="txt2img" store={store} />
    </QueryClientProvider>,
  )
  for (const [name, value] of Object.entries(values)) act(() => store.getState().setValue(OWNER, name, value))
  return store
}

const timing = (store: FormStore) => store.getState().values[OWNER]?.base_timing as Record<string, unknown>

async function panel() {
  return within(await screen.findByRole("group", { name: "Stage timing" }, { timeout: 40_000 }))
}

afterEach(() => vi.restoreAllMocks())

// The first test also loads the widget chunk lazily
describe("stage-timeline", { timeout: 60_000 }, () => {
  it("draws a template's parts and turns auto timing off at the window shown", async () => {
    const store = renderPanel()
    const group = await panel()
    expect(group.getByText(/runs the template's parts/)).toBeInTheDocument()
    expect(group.queryByRole("slider", { name: "Start ratio" })).toBeNull()  // the parts bring their own windows
    fireEvent.click(group.getByRole("switch", { name: "Auto timing from Shift" }))
    const style = stageWindow("style", 3)
    expect(timing(store)).toMatchObject({ auto: false, start: Number(style.start.toFixed(4)), end: 1, peak: Number(style.peak.toFixed(4)) })
    expect(group.getByRole("slider", { name: "Start ratio" })).toHaveAttribute("aria-valuenow", "0.62")
  })

  it("moves the window handles from the keyboard and keeps them in order", async () => {
    const store = renderPanel({ base_timing: { auto: false, start: 0.2, end: 0.6, peak: 0.4, strength: 1, curve: SMOOTH } })
    const group = await panel()
    const start = group.getByRole("slider", { name: "Start ratio" })
    fireEvent.keyDown(start, { key: "ArrowRight" })
    expect(timing(store)).toMatchObject({ start: 0.21, end: 0.6, peak: 0.4 })
    fireEvent.keyDown(start, { key: "End" })
    expect(timing(store)).toMatchObject({ start: 0.6, end: 0.6, peak: 0.6 })  // never past the end
    fireEvent.keyDown(group.getByRole("slider", { name: "End ratio" }), { key: "ArrowRight", shiftKey: true })
    expect(timing(store).end).toBe(0.65)
  })

  it("follows the generation form's Shift with auto timing", async () => {
    const store = renderPanel({ base_template: "partless" })
    const group = await panel()
    const startBox = () => group.getByRole("textbox", { name: "Start ratio" })
    expect(startBox()).toHaveValue(String(Number(stageWindow("style", 3).start.toFixed(4))))
    act(() => store.getState().setValue("core", "distilled_cfg_scale", 9))
    await waitFor(() => expect(startBox()).toHaveValue(String(Number(stageWindow("style", 9).start.toFixed(4)))))
    expect(group.getByText("Shift 9")).toBeInTheDocument()
    expect(timing(store).auto).toBe(true)  // display only: the run re-times it from its own Shift
    // An img2img Shift of 0 runs as 1 (Python clamps to 1-24), not as the default 3
    act(() => store.getState().setValue("core", "distilled_cfg_scale", 0))
    await waitFor(() => expect(group.getByText("Shift 1")).toBeInTheDocument())
    expect(startBox()).toHaveValue(String(Number(stageWindow("style", 1).start.toFixed(4))))
  })

  it("copies the template name without any page script", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true })
    renderPanel()
    const group = await panel()
    fireEvent.click(group.getByRole("button", { name: "Copy the template name" }))
    expect(writeText).toHaveBeenCalledWith(STYLE_ONLY)
  })

  it("combines presets into a template through the panel's action", async () => {
    const saved = { ...fixture.templates, mine: { label: "mine", builtin: false, auto: true, stage: "画风阶段", curve: null, parts: [] } }
    const action = vi.spyOn(api, "action").mockResolvedValue({ __toast__: { level: "success", message: "Saved" }, __outputs__: { templates: saved } } as never)
    renderPanel()
    const group = await panel()
    fireEvent.click(group.getByRole("button", { name: "Templates…" }))
    const dialog = within(await screen.findByRole("dialog"))
    // It starts on the panel template's parts (style applied)
    expect(dialog.getByRole("checkbox", { name: "Style stage only" })).toBeChecked()
    fireEvent.click(dialog.getByRole("checkbox", { name: "Block the style stage" }))
    expect(dialog.getByRole("alert")).toHaveTextContent("cannot be both applied and blocked")
    expect(dialog.getByRole("button", { name: "Save combo template" })).toBeDisabled()
    fireEvent.click(dialog.getByRole("checkbox", { name: "Block the style stage" }))
    fireEvent.click(dialog.getByRole("checkbox", { name: "Character stage only" }))
    fireEvent.change(dialog.getByRole("textbox", { name: "Template name" }), { target: { value: "mine" } })
    fireEvent.click(dialog.getByRole("button", { name: "Save combo template" }))
    await waitFor(() => expect(action).toHaveBeenCalled())
    const [owner, name, body] = action.mock.calls[0]!
    expect([owner, name]).toEqual([OWNER, "template_combo"])
    expect(body).toMatchObject({ name: "mine", presets: [STYLE_ONLY, "仅人物特征阶段介入"] })

    // The refreshed templates reach the editor
    fireEvent.click(dialog.getByRole("tab", { name: "Manage templates" }))
    expect(await dialog.findByRole("option", { name: "mine" })).toBeInTheDocument()
    fireEvent.click(dialog.getByRole("button", { name: "Delete template" }))
    await waitFor(() => expect(action).toHaveBeenLastCalledWith(OWNER, "template_delete", expect.objectContaining({ name: expect.any(String) })))
  })

  it("translates its chrome", () => {
    const keys = [...widgetKeys(source), ...widgetKeys(editorSource)]
    expect(keys.length).toBeGreaterThan(30)
    expect(keys.filter((key) => !(key in zh))).toEqual([])
  })
})
