/**
 * The img2img integration against the real schema: the source panel (mode
 * picker, source lines, mode options), the writes (new source clears the
 * painted layers, size from the image), the stage view switches and the
 * submit waiting for the canvas uploads.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import img2img from "@/api/mock/contract/img2img.json"
import { useProgress } from "@/api/progress"
import { keys, queryClient } from "@/api/queries"
import { dispatchRun, resetRunControllers, useRuns } from "@/features/progress/run-controller"
import { curateCore } from "@/pages/generate/curate"
import { loadSchema } from "@/pages/generate/use-form-loader"
import { FormProvider } from "@/ui/context"
import { getFormStore } from "@/ui/form-store"
import { registerPendingWrite } from "@/ui/pending-writes"
import { walk } from "@/ui/form-index"
import type { TabSchema, TabsNode } from "@/ui/schema"
import { pruneSource } from "./source-layout"
import { Img2imgSourcePanel } from "./SourcePanel"
import { useImg2imgStageView, useStageView } from "./stage-view"
import { setSizeFromImage, setSourceImage } from "./use-img2img"
import { editorBinding } from "../core/modes"
import { setPainting } from "../state/activity"
import { rememberImageSize } from "./image-size"

const toastSpy = vi.hoisted(() => vi.fn())
vi.mock("sonner", async (importOriginal) => {
  const actual = await importOriginal<typeof import("sonner")>()
  return { ...actual, toast: Object.assign((...args: unknown[]) => toastSpy(...args), actual.toast) }
})

const schema = img2img as unknown as TabSchema
const store = () => getFormStore("img2img")
const core = () => store().getState().values.core ?? {}

function renderPanel() {
  const client = new QueryClient({ defaultOptions: { queries: { enabled: false } } })
  client.setQueryData(keys.schema("img2img"), schema)
  const nodes = curateCore(schema.core.layout)
    .blocks.filter((block) => block.kind === "source")
    .map((block) => block.node)
  return render(
    <QueryClientProvider client={client}>
      <FormProvider formId="img2img" store={store()}>
        <Img2imgSourcePanel nodes={nodes} />
      </FormProvider>
    </QueryClientProvider>,
  )
}

const modes = () => screen.getByRole("radiogroup", { name: "Mode" })

// The panel renders the whole real schema: generous timeouts for loaded CI machines.
describe("img2img source panel (real schema)", { timeout: 30_000 }, () => {
  beforeEach(() => {
    store().setState({ revision: 0 })
    loadSchema(store(), schema)
  })

  it("offers every mode as one radio group and keeps the image inputs for the stage", () => {
    renderPanel()
    const radios = within(modes()).getAllByRole("radio")
    expect(radios.map((radio) => radio.textContent)).toEqual(["img2img", "Sketch", "Inpaint", "Inpaint sketch", "Inpaint upload", "Batch"])
    expect(within(modes()).getByRole("radio", { name: "img2img" })).toBeChecked()
    // No tab strip for the modes, no upload boxes for the canvas layers.
    expect(screen.queryByRole("tablist")).toBeNull()
    expect(screen.queryByText("Drop an image, paste, or click to browse")).toBeNull()
    expect(screen.getByText("Source image")).toBeInTheDocument()
    expect(screen.getByText("Resize Mode")).toBeInTheDocument()
  })

  it("shows each mode's options", () => {
    renderPanel()
    fireEvent.click(within(modes()).getByRole("radio", { name: "Inpaint" }))
    expect(core().mode).toBe("Inpaint")
    expect(screen.getByText("Paint the mask on the stage.")).toBeInTheDocument()
    expect(screen.getByText("Mask blur")).toBeInTheDocument()

    fireEvent.click(within(modes()).getByRole("radio", { name: "Inpaint upload" }))
    expect(screen.getByRole("button", { name: "Upload Mask" })).toBeInTheDocument()

    fireEvent.click(within(modes()).getByRole("radio", { name: "Batch" }))
    expect(screen.queryByText("Source image")).toBeNull()
    expect(screen.queryByText("Mask blur")).toBeNull()
    expect(screen.getByText("Append PNG Info")).toBeInTheDocument()
  })

  it("lays the mode out in balanced rows, labels whole", () => {
    renderPanel()
    // Six modes: three per row in a narrow pane (never four and two), six in a wide one.
    expect(modes().style.getPropertyValue("--cols")).toBe("3")
    expect(modes().style.getPropertyValue("--cols-wide")).toBe("6")
    expect(within(modes()).getByRole("radio", { name: "Inpaint sketch" })).toHaveTextContent("Inpaint sketch")
  })

  it("puts the source options straight in the card: no nested frames, hidden inputs or empty boxes", () => {
    const found = { tabs: null as TabsNode | null }
    const nodes = curateCore(schema.core.layout)
      .blocks.filter((block) => block.kind === "source")
      .map((block) => block.node)
    const handled = new Set(["init_img", "sketch", "mask", "mask_upload"])
    const flat = pruneSource(nodes, handled, "mode", found)
    expect(found.tabs?.bind).toBe("mode")
    const types: string[] = []
    for (const node of flat) walk(node, (n) => void types.push(n.type))
    expect(types).not.toContain("group")
    const names = flat.flatMap((node) => ("name" in node ? [node.name] : []))
    expect(names).toContain("resize_mode")
    expect(names).not.toContain("mode")
    // The inpaint options keep their condition on a plain column.
    const inpaint = flat.find((node) => node.type === "column" && JSON.stringify(node).includes("mask_blur"))
    expect(inpaint?.visible_if).toMatchObject({ param: "mode" })

    const { container } = renderPanel()
    expect(container.querySelector(".border-border\\/70")).toBeNull()
  })

  it("shows the size Resize by gives, snapped like the server", () => {
    rememberImageSize("src-1", { width: 1000, height: 750 })
    store().getState().setValues("core", { init_img: "src-1", resize_tab: "Resize by", scale_by: 1.5 })
    renderPanel()
    const step = (store().getState().index.inputs.get("core")?.get("width") as { step: number } | undefined)?.step ?? 8
    const snap = (v: number) => Math.floor(v / step + 0.5) * step
    expect(store().getState().outputs.core?.scale_by_resolution).toBe(`Resize from 1000×750 to ${snap(1500)}×${snap(1125)}`)
    act(() => store().getState().setValues("core", { scale_by: 2 }))
    expect(store().getState().outputs.core?.scale_by_resolution).toBe(`Resize from 1000×750 to ${snap(2000)}×${snap(1500)}`)
    rememberImageSize("src-1", null)
  })

  it("shows the Resize by size of a sent result from its output metadata, without reading the image", async () => {
    const step = (store().getState().index.inputs.get("core")?.get("width") as { step: number } | undefined)?.step ?? 8
    const snap = (v: number) => Math.floor(v / step + 0.5) * step
    const images = vi.spyOn(globalThis, "Image")
    store().getState().setValues("core", { resize_tab: "Resize by", scale_by: 1.5 })
    renderPanel()
    // As "Send to img2img" from a result (an output id, not an upload of this tab) does.
    const { sendTo } = await import("@/features/send-to/send-to")
    await act(() => sendTo("img2img", { fileId: "out-sent", width: 832, height: 1216 }, {}))
    expect(core().init_img).toBe("out-sent")
    expect(store().getState().outputs.core?.scale_by_resolution).toBe(`Resize from 832×1216 to ${snap(1248)}×${snap(1824)}`)
    expect(images).not.toHaveBeenCalled()
  })

  it("reads the size of a source without metadata once, for every reader, even when the first one unmounts", async () => {
    const loads: { src: string; onload: (() => void) | null }[] = []
    class FakeImage {
      naturalWidth = 640
      naturalHeight = 480
      onload: (() => void) | null = null
      onerror: (() => void) | null = null
      set src(value: string) {
        loads.push({ src: value, onload: () => this.onload?.() })
      }
    }
    vi.stubGlobal("Image", FakeImage)
    try {
      store().getState().setValues("core", { init_img: "out-unknown", resize_tab: "Resize by", scale_by: 2 })
      const first = renderPanel()
      first.unmount()
      renderPanel()
      expect(loads).toHaveLength(1)
      expect(loads[0]?.src).toContain("/files/out-unknown")
      act(() => loads[0]?.onload?.())
      expect(store().getState().outputs.core?.scale_by_resolution).toMatch(/^Resize from 640×480 to /)
    } finally {
      vi.unstubAllGlobals()
      rememberImageSize("out-unknown", null)
    }
  })

  it("shows the server's message for a missing mask", () => {
    store().getState().setValues("core", { mode: "Inpaint", init_img: "img-1" })
    store().getState().setErrors([{ owner: "core", param: "mask", message: "paint a mask over the image" }])
    renderPanel()
    expect(screen.getByRole("alert")).toHaveTextContent("paint a mask over the image")
    expect(screen.getByRole("button", { name: "Open the editor" })).toBeInTheDocument()
  })
})

describe("img2img writes", () => {
  beforeEach(() => {
    store().setState({ revision: 0 })
    loadSchema(store(), schema)
  })

  it("a new source image clears the layers painted over the old one, not the uploaded mask, undoably", async () => {
    toastSpy.mockClear()
    const binding = editorBinding(schema.core.layout)!
    store().getState().setValues("core", { init_img: "a", sketch: "s", mask: "m", mask_upload: "u" })
    await setSourceImage(binding, "b")
    expect(core()).toMatchObject({ init_img: "b", sketch: null, mask: null, mask_upload: "u" })
    expect(toastSpy).toHaveBeenCalledTimes(1)
    const [message, options] = toastSpy.mock.calls[0] as [string, { action: { label: string; onClick: () => void } }]
    expect(message).toBe("Image replaced")
    expect(options.action.label).toBe("Undo")
    options.action.onClick()
    expect(core()).toMatchObject({ init_img: "a", sketch: "s", mask: "m", mask_upload: "u" })
  })

  it("waits for the editor's last strokes before replacing, and asks nothing when there was no paint", async () => {
    toastSpy.mockClear()
    const binding = editorBinding(schema.core.layout)!
    store().getState().setValues("core", { init_img: "a", sketch: null, mask: null })
    const unregister = registerPendingWrite("img2img", async () => {
      store().getState().setValues("core", { mask: "late-stroke" })
    })
    await setSourceImage(binding, null)
    unregister()
    expect(core()).toMatchObject({ init_img: null, mask: null })
    // The stroke reached the form first, so the removal can be undone with it.
    expect(toastSpy).toHaveBeenCalledTimes(1)
    expect(toastSpy.mock.calls[0]![0]).toBe("Image removed")

    toastSpy.mockClear()
    store().getState().setValues("core", { init_img: "c", sketch: null, mask: null })
    await setSourceImage(binding, "d")
    expect(toastSpy).not.toHaveBeenCalled()
  })

  it("takes the image size on the sliders' grid and selects Resize to", () => {
    store().getState().setValues("core", { resize_tab: "Resize by" })
    setSizeFromImage({ width: 1001, height: 749 })
    const width = store().getState().index.inputs.get("core")?.get("width") as { step: number } | undefined
    const step = width?.step ?? 8
    expect(Number(core().width) % step).toBe(0)
    expect(core()).toMatchObject({ resize_tab: "Resize to" })
    expect(Math.abs(Number(core().width) - 1001)).toBeLessThanOrEqual(step / 2)
    expect(Math.abs(Number(core().height) - 749)).toBeLessThanOrEqual(step / 2)
  })
})

describe("img2img stage view", () => {
  beforeEach(() => {
    useStageView.setState({ view: "source" })
    useProgress.setState({ running: null, owned: {}, last: {}, orphans: {} })
    loadSchema(store(), schema)
  })

  function View() {
    const [view] = useImg2imgStageView(["init_img", "sketch", "mask", "mask_upload"])
    return <span data-testid="view">{view}</span>
  }

  it("turns to the result for the first job of a run only, never while the user paints", () => {
    render(<View />)
    const view = () => screen.getByTestId("view").textContent
    const run = (jobs: string[], extra: Record<string, unknown> = {}) =>
      useRuns.setState((s) => ({ runs: { ...s.runs, img2img: { ...s.runs.img2img, jobs, ...extra } } }))
    act(() => {
      run(["j1", "j2"])
      useProgress.setState({ owned: { j1: "img2img", j2: "img2img" }, running: { jobId: "j1", step: 0, steps: 20 } as never })
    })
    expect(view()).toBe("result")
    // Back to the editor to paint the next mask: the queued job starting does not pull the user out.
    act(() => useStageView.getState().setView("source"))
    act(() => {
      run(["j2"])
      useProgress.setState({ running: { jobId: "j2", step: 0, steps: 20 } as never })
    })
    expect(view()).toBe("source")
    // The run ends; the next one starts while a stroke is in progress: still no switch.
    act(() => {
      run([])
      useProgress.setState({ running: null })
    })
    const owner = {}
    setPainting(owner, true)
    act(() => {
      run(["j3"])
      useProgress.setState({ owned: { j3: "img2img" }, running: { jobId: "j3", step: 0, steps: 20 } as never })
    })
    setPainting(owner, false)
    expect(view()).toBe("source")
    act(() => {
      run([])
      useProgress.setState({ running: null })
    })
    resetRunControllers()
  })

  it("shows the result while a job of this form runs and the source when a new image arrives", () => {
    render(<View />)
    const view = () => screen.getByTestId("view").textContent
    act(() => {
      useProgress.setState({ owned: { j1: "img2img" }, running: { jobId: "j1", step: 0, steps: 20 } as never })
    })
    expect(view()).toBe("result")
    act(() => store().getState().setValues("core", { init_img: "sent-from-gallery" }))
    expect(view()).toBe("source")
    act(() => useStageView.getState().setView("result"))
    act(() => store().getState().setErrors([{ owner: "core", param: "mask", message: "paint a mask over the image" }]))
    expect(view()).toBe("source")
  })
})

describe("submit", () => {
  beforeEach(() => {
    resetRunControllers()
    queryClient.setQueryData(keys.schema("img2img"), schema)
    loadSchema(store(), schema)
  })
  afterEach(() => vi.restoreAllMocks())

  it("waits for the canvas uploads, so the job carries the mask on screen", async () => {
    const create = vi.spyOn(api.jobs, "create").mockResolvedValue({ job_id: "j9", job_ids: ["j9"], seq: 1 })
    const unregister = registerPendingWrite("img2img", async () => {
      await new Promise((resolve) => setTimeout(resolve, 20))
      store().getState().setValues("core", { mask: "fresh-mask" })
    })
    store().getState().setValues("core", { mode: "Inpaint", init_img: "img", mask: null })
    dispatchRun("img2img", { type: "generate" })
    await vi.waitFor(() => expect(create).toHaveBeenCalled())
    expect(create.mock.calls[0]![0].core).toMatchObject({ mask: "fresh-mask" })
    unregister()
  })
})
