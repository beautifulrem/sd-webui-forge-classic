import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { ImageEditorProps } from "@/canvas"
import { tabsKey } from "@/ui/form-index"
import { createFormStore, type FormStore } from "@/ui/form-store"
import { TabForm } from "@/ui/SchemaForm"
import type { Json, TabSchema, TabsNode } from "@/ui/schema"
import { widgetKeys } from "../_lib/i18n"
import contract from "./contract.json"
import zh from "./locales/zh-CN.json"
import source from "./index.tsx?raw"
import poseSource from "./PosePanel.tsx?raw"
import { defaultUnit, PASTE_MARKER, type UnitValue } from "./unit"

// The editor itself (Konva) is tested in src/canvas; here it reports what it was given and fakes its events
vi.mock("@/canvas", () => ({
  ImageEditor: (props: ImageEditorProps) => {
    const role = props.docKey.split(":").at(-1)
    return (
      <div data-testid={`editor-${role}`} data-image={props.image ?? ""} data-layer={props.activeLayer ?? ""}>
        {props.noUpload ? null : <button onClick={() => props.onImageChange?.({ id: `${role}-upload`, width: 100, height: 60 })}>{`upload ${role}`}</button>}
        {props.activeLayer ? <button onClick={() => props.onLayerChange?.(props.activeLayer!, `${role}-${props.activeLayer}`)}>{`paint ${role}`}</button> : null}
        {props.onUseImageSize ? <button onClick={() => props.onUseImageSize?.({ width: 517, height: 300 })}>use image size</button> : null}
      </div>
    )
  },
}))

const LOAD = { timeout: 15_000 }
const OWNER = "controlnet"
const TITLES = contract.props.tabs

// The server ships the catalog once per tab, with the first unit
const { catalog: CATALOG, ...UNIT_PROPS } = contract.props

function unitNode(index: number, img2img: boolean) {
  const props = index === 0 ? { ...UNIT_PROPS, catalog: CATALOG } : UNIT_PROPS
  return {
    type: "custom",
    name: `unit_${index}`,
    label: "",
    default: defaultUnit(img2img) as unknown as Json,
    widget: "controlnet-unit",
    infotext: `ControlNet ${index}`,
    props: { ...(props as unknown as Record<string, Json>), unit: index, img2img },
  } as const
}

function schemaFor(tab: "txt2img" | "img2img"): TabSchema {
  const img2img = tab === "img2img"
  const units = TITLES.map((_, i) => unitNode(i, img2img))
  return {
    version: 1,
    tab,
    core: {
      owner: "core",
      layout: {
        type: "column",
        children: [
          { type: "slider", name: "width", label: "Width", default: 512, minimum: 64, maximum: 2048, step: 8, integer: true },
          { type: "slider", name: "height", label: "Height", default: 512, minimum: 64, maximum: 2048, step: 8, integer: true },
          { type: "toggle", name: "enable_hr", label: "Hires. fix", default: false },
          { type: "slot", section: "scripts" },
        ],
      },
      defaults: { width: 512, height: 512, enable_hr: false },
    },
    scripts: [
      {
        id: OWNER,
        title: "ControlNet",
        kind: "alwayson",
        section: null,
        order: 0,
        layout: {
          type: "section",
          title: "ControlNet Integrated",
          open: true,
          children: [{ type: "tabs", children: TITLES.map((title, i) => ({ type: "tab", title, children: [units[i]!] })) }],
        },
        defaults: Object.fromEntries(units.map((unit) => [unit.name, unit.default])),
        args: units.map((unit) => unit.name),
      },
    ],
  }
}

function renderForm(tab: "txt2img" | "img2img" = "txt2img"): FormStore {
  const store = createFormStore(`controlnet-${Math.random()}`, { persistDrafts: false })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, enabled: false } } })
  render(
    <QueryClientProvider client={client}>
      <TabForm schema={schemaFor(tab)} formId={tab} store={store} />
    </QueryClientProvider>,
  )
  return store
}

const unitOf = (store: FormStore, index = 0) => store.getState().values[OWNER]?.[`unit_${index}`] as UnitValue
const setUnit = (store: FormStore, patch: Partial<UnitValue>, index = 0) =>
  act(() => store.getState().setValues(OWNER, { [`unit_${index}`]: { ...unitOf(store, index), ...patch } }))

type Answer = (action: string, body: Record<string, unknown>) => unknown

/** Answers `POST /actions/controlnet/<action>`; returns the request bodies. */
function mockActions(answer: Answer) {
  const bodies: [string, Record<string, unknown>][] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const request = input instanceof Request ? input : new Request(String(input), init)
    const action = decodeURIComponent(new URL(request.url).pathname.split("/").at(-1) ?? "")
    const body = (await request.json()) as Record<string, unknown>
    bodies.push([action, body])
    return new Response(JSON.stringify(answer(action, body) ?? {}), { status: 200, headers: { "Content-Type": "application/json" } })
  })
  return bodies
}

/** The first unit's panel (the tab strip shows one unit at a time). */
const panel = async () => screen.findByRole("group", { name: TITLES[0] }, LOAD)

describe("controlnet-unit widget", { timeout: 30_000 }, () => {
  afterEach(() => vi.restoreAllMocks())

  it("switches and type filtering write the unit", async () => {
    const store = renderForm()
    const unit = await panel()
    fireEvent.click(within(unit).getByRole("switch", { name: "Enable" }))
    expect(unitOf(store).enabled).toBe(true)

    fireEvent.click(within(unit).getByRole("radio", { name: "Canny" }))
    expect(unitOf(store)).toMatchObject({ type_filter: "Canny", module: "canny", model: "control_v11p_sd15_canny", threshold_a: 100, threshold_b: 200, processor_res: 512 })
    // canny's own sliders replace "Threshold A/B"
    expect(within(unit).getByRole("textbox", { name: "Low Threshold" })).toHaveValue("100")
    expect(within(unit).getByRole("textbox", { name: "Resolution" })).toHaveValue("512")
    expect(within(unit).queryByRole("textbox", { name: "Threshold A" })).not.toBeInTheDocument()

    fireEvent.click(within(unit).getByRole("switch", { name: "Pixel Perfect" }))
    expect(within(unit).queryByRole("textbox", { name: "Resolution" })).not.toBeInTheDocument()

    const weight = within(unit).getByRole("textbox", { name: "Control Weight" })
    fireEvent.change(weight, { target: { value: "1.25" } })
    fireEvent.blur(weight)
    expect(unitOf(store).weight).toBe(1.25)
    const from = within(unit).getByRole("textbox", { name: "Timestep Range: from" })
    fireEvent.change(from, { target: { value: "0.2" } })
    fireEvent.blur(from)
    expect(unitOf(store)).toMatchObject({ guidance_start: 0.2, guidance_end: 1 })

    fireEvent.click(within(unit).getByRole("radio", { name: "My prompt is more important" }))
    expect(unitOf(store).control_mode).toBe("My prompt is more important")
  })

  it("edits the control image, its layers and the generation size", async () => {
    const store = renderForm()
    const unit = await panel()
    fireEvent.click(within(unit).getByRole("button", { name: "upload input" }))
    // A new control image switches the unit on, as the old upload did
    expect(unitOf(store)).toMatchObject({ image: "input-upload", enabled: true })
    fireEvent.click(within(unit).getByRole("switch", { name: "Enable" }))
    fireEvent.click(within(unit).getByRole("button", { name: "upload input" }))
    expect(unitOf(store).enabled).toBe(true)
    expect(within(unit).getByTestId("editor-input")).toHaveAttribute("data-layer", "image_fg")
    fireEvent.click(within(unit).getByRole("button", { name: "paint input" }))
    expect(unitOf(store).image_fg).toBe("input-image_fg")

    fireEvent.click(within(unit).getByRole("radio", { name: "Sketch" }))
    fireEvent.click(within(unit).getByRole("button", { name: "paint input" }))
    expect(unitOf(store).sketch).toBe("input-sketch")

    fireEvent.click(within(unit).getByRole("button", { name: "use image size" }))
    expect(store.getState().values.core).toMatchObject({ width: 520, height: 296 })

    // A new image drops the layers painted over the old one
    fireEvent.click(within(unit).getByRole("button", { name: "upload input" }))
    expect(unitOf(store)).toMatchObject({ sketch: null, image_fg: null })

    fireEvent.click(within(unit).getByRole("switch", { name: "Use Mask" }))
    const mask = within(unit).getByTestId("editor-mask")
    expect(mask).toHaveAttribute("data-image", "input-upload") // painted over the control image until a mask is uploaded
    fireEvent.click(within(mask).getByRole("button", { name: "paint mask" }))
    expect(unitOf(store)).toMatchObject({ mask_upload: true, mask_image_fg: "mask-mask_image_fg" })
    fireEvent.click(within(unit).getByRole("switch", { name: "Use Mask" }))
    expect(unitOf(store)).toMatchObject({ mask_upload: false, mask_image_fg: null })
  })

  it("runs the preprocessor preview and uses it as the input", async () => {
    const bodies = mockActions((action, body) => (action === "preview" ? { __outputs__: { preview: { unit: body.unit, image: "preview-1", pose: null } } } : {}))
    const store = renderForm()
    const unit = await panel()
    fireEvent.click(within(unit).getByRole("button", { name: "upload input" }))
    expect(unitOf(store).enabled).toBe(true)
    // Edits made while the preview runs are kept
    fireEvent.click(within(unit).getByRole("button", { name: "Run preprocessor" }))
    fireEvent.click(within(unit).getByRole("switch", { name: "Enable" }))
    await waitFor(() => expect(unitOf(store).generated_image).toBe("preview-1"))
    expect(unitOf(store).enabled).toBe(false)
    expect(bodies[0]![0]).toBe("preview")
    expect(bodies[0]![1]).toMatchObject({ unit: 0, tab: "txt2img" })

    expect(within(unit).getByTestId("editor-preview")).toHaveAttribute("data-image", "preview-1")
    fireEvent.click(within(unit).getByRole("switch", { name: "Preview as Input" }))
    expect(unitOf(store).use_preview_as_input).toBe(true)
    expect(within(unit).getByRole("button", { name: "Download preview" })).toHaveAttribute("href", expect.stringContaining("preview-1"))
    fireEvent.click(within(unit).getByRole("button", { name: "Close preview" }))
    expect(unitOf(store)).toMatchObject({ generated_image: null, use_preview_as_input: false })
  })

  it("applies an edited pose", async () => {
    const pose = { canvas_width: 64, canvas_height: 32, people: [] }
    const bodies = mockActions((action, body) => (action === "render_pose" ? { __outputs__: { pose: { unit: body.unit, image: "pose-1", pose: body.pose } } } : {}))
    const store = renderForm()
    const unit = await panel()
    setUnit(store, { type_filter: "OpenPose" })
    const editor = within(unit).getByRole("textbox", { name: "Pose (OpenPose JSON)" })
    fireEvent.change(editor, { target: { value: "{" } })
    fireEvent.click(within(unit).getByRole("button", { name: "Apply pose" }))
    expect(within(unit).getByText(/Not a valid pose/)).toBeInTheDocument()
    expect(bodies).toHaveLength(0)

    fireEvent.change(editor, { target: { value: JSON.stringify(pose) } })
    fireEvent.click(within(unit).getByRole("button", { name: "Apply pose" }))
    await waitFor(() => expect(unitOf(store)).toMatchObject({ generated_image: "pose-1", use_preview_as_input: true, pose }))
  })

  it("generates the Region canvas", async () => {
    const bodies = mockActions((action, body) => (action === "region_canvas" ? { __outputs__: { canvas: { unit: body.unit, image: "region-1" } } } : {}))
    const store = renderForm()
    const unit = await panel()
    fireEvent.click(within(unit).getByRole("radio", { name: "Region" }))
    expect(within(unit).getByTestId("editor-input")).toHaveAttribute("data-layer", "sketch")
    fireEvent.click(within(unit).getByRole("radio", { name: "Oval" }))
    fireEvent.click(within(unit).getByRole("button", { name: "Generate Canvas" }))
    await waitFor(() => expect(unitOf(store).image).toBe("region-1"))
    expect(bodies[0]![1]).toMatchObject({ unit: 0, shape: "Oval", subjects: 0, vertical: 0.4, randomize: 0.075 })
  })

  it("refreshes the model list for every unit", async () => {
    const models = ["None", "brand_new_model"]
    const catalog = { ...contract.props.catalog, models, types: contract.props.catalog.types.map((type) => (type.name === "All" ? { ...type, models } : type)) }
    mockActions((action) => (action === "refresh_models" ? { __outputs__: { catalog } } : {}))
    renderForm()
    const unit = await panel()
    fireEvent.click(within(unit).getByRole("button", { name: "Refresh models" }))
    const user = userEvent.setup()
    await waitFor(() => expect(fetch).toHaveBeenCalled())
    await user.click(within(unit).getByRole("combobox", { name: "Model" }))
    expect(await screen.findByRole("option", { name: "brand_new_model" })).toBeInTheDocument()
  })

  it("reads the catalog shipped with the first unit in every unit", async () => {
    const store = renderForm()
    await panel()
    const key = tabsKey(OWNER, { children: TITLES.map((title) => ({ title })) } as TabsNode)
    act(() => store.getState().setTab(key, "2"))
    const third = await screen.findByRole("group", { name: TITLES[2] }, LOAD)
    fireEvent.click(within(third).getByRole("radio", { name: "Canny" }))
    expect(unitOf(store, 2)).toMatchObject({ type_filter: "Canny", module: "canny", model: "control_v11p_sd15_canny" })
    // A paste into a unit without a catalog of its own still resolves its control type
    act(() => store.getState().setValues(OWNER, { unit_1: { [PASTE_MARKER]: true, module: "canny" } as Json }))
    expect(unitOf(store, 1)).toMatchObject({ module: "canny", type_filter: "All" })
  })

  it("counts the enabled units in the section header", async () => {
    const store = renderForm()
    await panel()
    const header = screen.getByRole("button", { name: /^ControlNet Integrated/, expanded: true })
    expect(header).not.toHaveTextContent("Unit")
    setUnit(store, { enabled: true })
    setUnit(store, { enabled: true }, 2)
    expect(header).toHaveTextContent("2x Unit")
    // Shown while the section is collapsed too
    fireEvent.click(header)
    expect(header).toHaveAttribute("aria-expanded", "false")
    expect(header).toHaveTextContent("2x Unit")
    setUnit(store, { enabled: false })
    expect(header).toHaveTextContent("1x Unit")
  })

  it("shows the Hires. fix option only with Hires. fix (txt2img)", async () => {
    const store = renderForm()
    const unit = await panel()
    expect(within(unit).queryByRole("radiogroup", { name: "Hires-Fix Option" })).not.toBeInTheDocument()
    act(() => store.getState().setValues("core", { enable_hr: true }))
    expect(within(unit).getByRole("radiogroup", { name: "Hires-Fix Option" })).toBeInTheDocument()
  })

  it("img2img units read the img2img image unless independent", async () => {
    const store = renderForm("img2img")
    const unit = await panel()
    expect(within(unit).queryByTestId("editor-input")).not.toBeInTheDocument()
    expect(within(unit).queryByRole("radiogroup", { name: "Resize Mode" })).not.toBeInTheDocument()
    fireEvent.click(within(unit).getByRole("switch", { name: "Upload independent control image" }))
    fireEvent.click(within(unit).getByRole("button", { name: "upload input" }))
    expect(unitOf(store)).toMatchObject({ upload_independent: true, image: "input-upload" })
    fireEvent.click(within(unit).getByRole("switch", { name: "Upload independent control image" }))
    expect(unitOf(store)).toMatchObject({ upload_independent: false, image: null })
  })

  it("a paste keeps the unit's images and switch", async () => {
    const store = renderForm()
    await panel()
    setUnit(store, { enabled: true, image: "kept", image_fg: "mask" })
    act(() => store.getState().setValues(OWNER, { unit_0: { [PASTE_MARKER]: true, module: "canny", weight: 0.3 } as Json }))
    expect(unitOf(store)).toMatchObject({ enabled: true, image: "kept", image_fg: "mask", module: "canny", weight: 0.3 })
    expect(PASTE_MARKER in unitOf(store)).toBe(false)
    // The infotext leaves an unset (-1) resolution out; the server pastes the default, which resets it
    setUnit(store, { processor_res: 768, threshold_a: 50 })
    act(() => store.getState().setValues(OWNER, { unit_0: { [PASTE_MARKER]: true, module: "canny", processor_res: -1, threshold_a: -1 } as Json }))
    expect(unitOf(store)).toMatchObject({ processor_res: -1, threshold_a: -1, image: "kept" })
  })

  it("lists the enabled units and switches to one", async () => {
    const store = renderForm()
    const unit = await panel()
    expect(within(unit).queryByText("Enabled units:")).not.toBeInTheDocument()
    setUnit(store, { enabled: true, type_filter: "Canny" }, 2)
    const chip = within(unit).getByRole("button", { name: /ControlNet Unit 3/ })
    expect(chip).toHaveTextContent("Canny")
    fireEvent.click(chip)
    const key = tabsKey(OWNER, { children: TITLES.map((title) => ({ title })) } as TabsNode)
    expect(store.getState().tabs[key]).toBe("2")
  })

  it("translates every chrome string", () => {
    const missing = [...widgetKeys(source), ...widgetKeys(poseSource), "{{count}}x Unit"].filter((key) => !(key in zh))
    expect(missing).toEqual([])
  })
})
