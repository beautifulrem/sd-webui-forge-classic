import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import type { PageInfo } from "@/api/types"
import { navigate } from "@/app/router"
import { getFormStore } from "@/ui/form-store"
import { tabOwners } from "@/ui/tab"
import { toast } from "sonner"
import { finalPrompt } from "./draft"
import JsonBuilderPage from "./index"
import { sizeLimit } from "./size-limit"
import { useBuilder } from "./store"
import { COMPILED, DRAFT } from "./testdata"

const PAGE = { id: "json-builder", title: "JSON Builder", icon: "braces", widget: "json-builder" } as PageInfo
const initial = useBuilder.getState()

function renderPage() {
  return render(<JsonBuilderPage page={PAGE} />)
}

beforeEach(() => {
  useBuilder.setState(initial, true)
  navigate({ name: "page", id: "json-builder" }, { replace: true })
  // on_change actions of the forms (hires resolution) answer nothing
  vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response("{}", { headers: { "Content-Type": "application/json" } }))
})
afterEach(() => vi.restoreAllMocks())

describe("JSON Builder page", () => {
  it("shows the draft's items on the canvas and in the list", () => {
    renderPage()
    const canvas = screen.getByTestId("json-builder-canvas")
    expect(within(canvas).getAllByRole("group")).toHaveLength(1)
    fireEvent.click(screen.getByRole("button", { name: "Add item" }))
    expect(within(canvas).getAllByRole("group")).toHaveLength(2)
    expect(within(screen.getByRole("list", { name: "Items" })).getAllByRole("listitem")).toHaveLength(2)
    fireEvent.click(screen.getByRole("button", { name: "Remove item 1" }))
    expect(within(canvas).getAllByRole("group")).toHaveLength(1)
  })

  it("marks the empty fields instead of sending", () => {
    const load = getFormStore("txt2img").getState()
    load.load(tabOwners(tabSchema("txt2img")))
    renderPage()
    fireEvent.click(screen.getByRole("button", { name: "Send to txt2img" }))
    expect(screen.getByLabelText("Caption")).toHaveAttribute("aria-invalid", "true")
    expect(screen.getByLabelText("Caption of item 1")).toHaveAttribute("aria-invalid", "true")
    expect(location.hash).toBe("#/ext/json-builder")
    // a filled field is no longer marked
    fireEvent.change(screen.getByLabelText("Caption"), { target: { value: "a room" } })
    expect(screen.getByLabelText("Caption")).not.toHaveAttribute("aria-invalid")
  })

  it("sends the prompt, width and height to txt2img and shows it", async () => {
    getFormStore("txt2img").getState().load(tabOwners(tabSchema("txt2img")))
    renderPage()
    fireEvent.change(screen.getByLabelText("Caption"), { target: { value: "A cat (orange)" } })
    fireEvent.change(screen.getByLabelText("Caption of item 1"), { target: { value: "a cat" } })
    fireEvent.change(screen.getByLabelText("Extra"), { target: { value: "<lora:moss:1>" } })
    fireEvent.click(screen.getByRole("button", { name: "Send to txt2img" }))
    await waitFor(() => expect(location.hash).toBe("#/txt2img"))
    const core = getFormStore("txt2img").getState().values.core
    expect(core?.prompt).toBe(finalPrompt(useBuilder.getState()))
    expect(core?.prompt).toMatch(/^\{"summary":\{"canvas":"single","caption":"A cat \\\(orange\\\)"\}.*"bbox_format":"yxyx"\} <lora:moss:1>$/)
    expect([core?.width, core?.height]).toEqual([1024, 1024])
  })

  it("warns when the size is past what the target form accepts", async () => {
    const warning = vi.spyOn(toast, "warning")
    const success = vi.spyOn(toast, "success")
    for (const tab of ["txt2img", "img2img"] as const) getFormStore(tab).getState().load(tabOwners(tabSchema(tab)))
    useBuilder.getState().setSize(3072, 1024)
    renderPage()
    fireEvent.change(screen.getByLabelText("Caption"), { target: { value: "a room" } })
    fireEvent.change(screen.getByLabelText("Caption of item 1"), { target: { value: "a lamp" } })
    fireEvent.click(screen.getByRole("button", { name: "Send to img2img" }))
    await waitFor(() => expect(warning).toHaveBeenCalledOnce())
    expect(warning.mock.calls[0]![0]).toMatch(/at most 2048 px/)
    expect(success).not.toHaveBeenCalled()
    // the values still arrive: the form shows them and its check names the field
    expect(getFormStore("img2img").getState().values.core?.width).toBe(3072)
    // txt2img takes up to 4096
    expect(sizeLimit("txt2img", 3072, 1024)).toBeNull()
    expect(sizeLimit("img2img", 1024, 2048)).toBeNull()
    expect(sizeLimit("img2img", 1024, 2064)).toBe(2048)
  })

  it("loads a compiled prompt (and the text after it) for the size given", async () => {
    useBuilder.getState().setSize(1024, 768)
    renderPage()
    fireEvent.click(screen.getByRole("button", { name: "Load prompt…" }))
    const dialog = await screen.findByRole("dialog")
    fireEvent.change(within(dialog).getByLabelText("Compiled prompt or layout draft"), { target: { value: "not json" } })
    fireEvent.click(within(dialog).getByRole("button", { name: "Load" }))
    expect(within(dialog).getByRole("alert")).toHaveTextContent(/JSON/)

    fireEvent.change(within(dialog).getByLabelText("Compiled prompt or layout draft"), { target: { value: `${COMPILED} <lora:x:1>` } })
    fireEvent.click(within(dialog).getByRole("button", { name: "Load" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
    const s = useBuilder.getState()
    expect(s.items.map((item) => [item.label, item.caption, item.bbox])).toEqual([
      ["character", "an orange cat", [100, 200, 501, 700]],
      ["text", "a neon sign", [600, 50, 1000, 150]],
    ])
    expect([s.caption, s.extra]).toEqual(["A cat (orange) on a sofa", "<lora:x:1>"])
    expect(screen.getByLabelText("Caption of item 2")).toHaveValue("a neon sign")
  })

  it("loads a saved layout file", async () => {
    renderPage()
    const file = new File([JSON.stringify(DRAFT)], "layout.json", { type: "application/json" })
    fireEvent.change(screen.getByTestId("json-builder-file"), { target: { files: [file] } })
    await waitFor(() => expect(useBuilder.getState().items).toHaveLength(2))
    expect([useBuilder.getState().width, useBuilder.getState().height]).toEqual([1024, 768])
  })

  it("drags a box and its corner in canvas pixels, within the canvas", () => {
    // the stage is drawn at half size: one client pixel is two canvas pixels
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue(DOMRect.fromRect({ x: 0, y: 0, width: 512, height: 512 }))
    renderPage()
    const box = within(screen.getByTestId("json-builder-canvas")).getByRole("group")
    fireEvent.pointerDown(box, { button: 0, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(window, { clientX: 101, clientY: 100 }) // under the threshold
    expect(useBuilder.getState().items[0]!.bbox).toEqual([102, 102, 409, 409])
    fireEvent.pointerMove(window, { clientX: 150, clientY: 90 })
    fireEvent.pointerUp(window)
    expect(useBuilder.getState().items[0]!.bbox).toEqual([202, 82, 509, 389])
    fireEvent.pointerMove(window, { clientX: 400, clientY: 400 }) // released: no effect
    expect(useBuilder.getState().items[0]!.bbox).toEqual([202, 82, 509, 389])

    const handle = box.querySelector(".cursor-nwse-resize")!
    fireEvent.pointerDown(handle, { button: 0, clientX: 0, clientY: 0 })
    fireEvent.pointerMove(window, { clientX: 1000, clientY: -1000 })
    fireEvent.pointerUp(window)
    expect(useBuilder.getState().items[0]!.bbox).toEqual([202, 82, 1024, 98])
  })

  it("zooms with the buttons and the wheel, and fits again", () => {
    renderPage()
    const canvas = screen.getByTestId("json-builder-canvas")
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }))
    expect(canvas).toHaveTextContent("120%")
    fireEvent.wheel(canvas, { deltaY: 100 })
    fireEvent.wheel(canvas, { deltaY: 100 })
    expect(canvas).toHaveTextContent("83%")
    fireEvent.click(screen.getByRole("button", { name: "Fit to screen" }))
    expect(canvas).toHaveTextContent("100%")
  })

  it("edits a caption on the canvas after a double click", () => {
    renderPage()
    const field = screen.getByLabelText("Box 1 caption")
    expect(field).toHaveAttribute("readonly")
    fireEvent.doubleClick(field)
    expect(field).not.toHaveAttribute("readonly")
    fireEvent.change(field, { target: { value: "a lamp" } })
    expect(screen.getByLabelText("Caption of item 1")).toHaveValue("a lamp")
    fireEvent.blur(field)
    expect(field).toHaveAttribute("readonly")
  })

  it("moves the focused box with the arrow keys", () => {
    renderPage()
    const box = within(screen.getByTestId("json-builder-canvas")).getByRole("group")
    fireEvent.keyDown(box, { key: "ArrowRight" })
    fireEvent.keyDown(box, { key: "ArrowDown", shiftKey: true })
    expect(useBuilder.getState().items[0]!.bbox).toEqual([103, 118, 410, 425])
  })
})
