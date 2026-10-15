import { act, fireEvent, render, screen, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { FormProvider, OwnerProvider } from "@/ui/context"
import { createFormStore, type FormStore } from "@/ui/form-store"
import type { CustomNode, Node, OwnerDefaults } from "@/ui/schema"
import { widgetKeys } from "../_lib/i18n"
import zh from "./locales/zh-CN.json"
import { LAYOUTS } from "./geometry"
import RegionBoxes from "./index"

const OWNER = "anima_regional_conditioning"
const PREFIXES = ["region1", "region2", "region3"]
const DEFAULTS = [
  { active: true, prompt: "a red cat", x: 0, y: 0, width: 0.5, height: 1, strength: 1 },
  { active: true, prompt: "", x: 0.5, y: 0, width: 0.5, height: 1, strength: 1 },
  { active: false, prompt: "", x: 0.25, y: 0.25, width: 0.5, height: 0.5, strength: 1 },
]

const EDITOR: CustomNode = { type: "custom", name: "region_editor", label: "", widget: "region-boxes", props: { regions: PREFIXES }, default: null }

function layout(): Node {
  const children: Node[] = [EDITOR]
  for (const prefix of PREFIXES) {
    children.push({ type: "toggle", name: `${prefix}_active`, label: "Active" })
    children.push({ type: "text", name: `${prefix}_prompt`, label: "Regional prompt" })
    for (const field of ["x", "y", "width", "height"]) {
      children.push({ type: "slider", name: `${prefix}_${field}`, label: field, minimum: 0, maximum: 1, step: 0.01, integer: false })
    }
    children.push({ type: "slider", name: `${prefix}_strength`, label: "Strength", minimum: 0, maximum: 2, step: 0.05, integer: false })
  }
  return { type: "group", children }
}

function setup({ disabled = false, conflictParams = [] as string[] } = {}): FormStore {
  const store = createFormStore(`region-boxes-${Math.random()}`, { persistDrafts: false })
  const defaults: OwnerDefaults = { region_editor: null }
  DEFAULTS.forEach((region, index) => {
    for (const [field, value] of Object.entries(region)) defaults[`${PREFIXES[index]}_${field}`] = value
  })
  store.getState().load(
    [
      { owner: "core", layout: null, defaults: { width: 1216, height: 832 } },
      { owner: OWNER, layout: layout(), defaults },
    ],
    { conflictParams },
  )
  render(
    <FormProvider formId="txt2img" store={store}>
      <OwnerProvider owner={OWNER}>
        <RegionBoxes value={null} onChange={() => {}} props={{ regions: PREFIXES }} node={EDITOR} disabled={disabled} id="editor" label="" />
      </OwnerProvider>
    </FormProvider>,
  )
  return store
}

const values = (store: FormStore) => store.getState().values[OWNER] ?? {}

function canvasOf(width = 200, height = 100) {
  const canvas = screen.getByTestId("region-canvas")
  canvas.getBoundingClientRect = () => ({ x: 0, y: 0, left: 0, top: 0, right: width, bottom: height, width, height, toJSON: () => ({}) })
  return canvas
}

const region = (number: number) => screen.getByRole("group", { name: new RegExp(`^Region ${number}:`) })

afterEach(() => vi.restoreAllMocks())

// The first render (jsdom, Radix) can take seconds on a loaded machine
describe("region-boxes", { timeout: 20_000 }, () => {
  it("draws the active regions over an image-shaped preview", () => {
    setup()
    const canvas = canvasOf()
    expect(canvas.style.aspectRatio).toBe("1216 / 832")
    expect(region(1)).toHaveAccessibleName("Region 1: x 0.00, y 0.00, width 0.50, height 1.00")
    expect(region(1).style.left).toBe("0%")
    expect(region(2).style.left).toBe("50%")
    expect(within(region(1)).getByText("a red cat")).toBeInTheDocument()
    // An empty prompt is shown as unused; the inactive region is out of the tab order
    expect(within(region(2)).getByText("Empty prompt: not used")).toBeInTheDocument()
    expect(canvas.querySelector('[data-region="3"]')).toHaveAttribute("tabindex", "-1")
  })

  it("moves and resizes the focused region with the arrow keys", () => {
    const store = setup()
    fireEvent.keyDown(region(2), { key: "ArrowLeft" })
    expect(values(store).region2_x).toBe(0.49)
    fireEvent.keyDown(region(2), { key: "ArrowLeft", shiftKey: true })
    expect(values(store).region2_width).toBe(0.49)
    fireEvent.keyDown(region(2), { key: "ArrowDown", altKey: true })
    // Already full height: a move stays inside the image
    expect(values(store).region2_y).toBe(0)
  })

  it("keeps keyboard focus on a region when it becomes selected", () => {
    const store = setup()
    // Region 2 is not selected yet: focusing it must not move its node and drop focus
    act(() => region(2).focus())
    expect(document.activeElement).toBe(region(2))
    fireEvent.keyDown(document.activeElement!, { key: "ArrowLeft" })
    expect(values(store).region2_x).toBe(0.49)
    // The region list buttons focus the region too
    fireEvent.click(screen.getByRole("button", { name: "Region 1" }))
    expect(document.activeElement).toBe(region(1))
    // A click selects and focuses, and the DOM (tab) order stays the region order
    canvasOf()
    fireEvent.pointerDown(region(2), { pointerId: 1, button: 0, buttons: 1, clientX: 150, clientY: 50 })
    fireEvent.pointerUp(region(2), { pointerId: 1, clientX: 150, clientY: 50 })
    expect(document.activeElement).toBe(region(2))
    const order = [...screen.getByTestId("region-canvas").querySelectorAll("[data-region]")].map((node) => node.getAttribute("data-region"))
    expect(order).toEqual(["1", "2", "3"])
    expect(region(2)).toHaveAttribute("aria-keyshortcuts")
    expect(region(2)).toHaveAccessibleDescription(/Arrow keys move the region/)
  })

  it("ends a drag when the button is no longer pressed or the capture is lost", () => {
    const action = vi.spyOn(api, "action").mockResolvedValue({})
    const store = setup({ conflictParams: [`${OWNER}.region1_x`] })
    canvasOf(200, 100)
    const box = region(1)
    // The pointerup landed elsewhere: the next move without a pressed button ends the drag
    fireEvent.pointerDown(box, { pointerId: 1, button: 0, buttons: 1, clientX: 10, clientY: 10 })
    fireEvent.pointerMove(box, { pointerId: 1, buttons: 1, clientX: 30, clientY: 10, shiftKey: true })
    expect(values(store).region1_x).toBe(0.1)
    fireEvent.pointerMove(box, { pointerId: 1, buttons: 0, clientX: 60, clientY: 10, shiftKey: true })
    fireEvent.pointerMove(box, { pointerId: 1, buttons: 1, clientX: 90, clientY: 10, shiftKey: true })
    expect(values(store).region1_x).toBe(0.1)
    expect(action).toHaveBeenCalledTimes(1)
    // A lost capture ends it too
    fireEvent.pointerDown(box, { pointerId: 1, button: 0, buttons: 1, clientX: 10, clientY: 10 })
    fireEvent.lostPointerCapture(box, { pointerId: 1 })
    fireEvent.pointerMove(box, { pointerId: 1, buttons: 1, clientX: 90, clientY: 10, shiftKey: true })
    expect(values(store).region1_x).toBe(0.1)
  })

  it("drags a region and runs the change effects once, when the drag ends", () => {
    const action = vi.spyOn(api, "action").mockResolvedValue({})
    const store = setup({ conflictParams: [`${OWNER}.region1_x`] })
    canvasOf(200, 100)
    const box = region(1)
    fireEvent.pointerDown(box, { pointerId: 1, button: 0, clientX: 10, clientY: 10 })
    fireEvent.pointerMove(box, { pointerId: 1, buttons: 1, clientX: 30, clientY: 10 })
    fireEvent.pointerMove(box, { pointerId: 1, buttons: 1, clientX: 50, clientY: 10, shiftKey: true })
    expect(values(store).region1_x).toBe(0.2)
    expect(values(store).region1_width).toBe(0.5)
    expect(action).not.toHaveBeenCalled()
    fireEvent.pointerUp(box, { pointerId: 1, clientX: 50, clientY: 10 })
    expect(action).toHaveBeenCalledTimes(1)
    expect(action.mock.calls[0]?.[1]).toBe("__conflicts__")
  })

  it("resizes from a corner handle", () => {
    const store = setup()
    canvasOf(200, 100)
    fireEvent.focus(region(1))
    const handle = region(1).querySelector('[data-handle="se"]')!
    fireEvent.pointerDown(handle, { pointerId: 2, button: 0, clientX: 100, clientY: 100 })
    let writes = 0
    const unsubscribe = store.subscribe(() => writes++)
    fireEvent.pointerMove(handle, { pointerId: 2, buttons: 1, clientX: 80, clientY: 70, shiftKey: true })
    unsubscribe()
    // Handled once (by the region), not again on the way up from the handle
    expect(writes).toBe(1)
    fireEvent.pointerUp(handle, { pointerId: 2 })
    expect(values(store)).toMatchObject({ region1_x: 0, region1_y: 0, region1_width: 0.4, region1_height: 0.7 })
  })

  it("applies a layout and switches regions on and off", () => {
    const store = setup()
    fireEvent.click(screen.getByRole("button", { name: "Three columns" }))
    expect(values(store)).toMatchObject({ region3_active: true, region1_width: 0.33, region2_x: 0.33, region3_x: 0.67 })
    fireEvent.click(screen.getByRole("button", { name: "Two rows" }))
    expect(values(store)).toMatchObject({ region3_active: false, region1_height: 0.5, region2_y: 0.5 })
    act(() => {
      fireEvent.click(screen.getByRole("checkbox", { name: "Region 3 active" }))
    })
    expect(values(store).region3_active).toBe(true)
  })

  it("follows the sliders: the params are the source of truth", () => {
    const store = setup()
    act(() => store.getState().setValues(OWNER, { region1_x: 0.1, region1_width: 0.3 }))
    expect(region(1)).toHaveAccessibleName("Region 1: x 0.10, y 0.00, width 0.30, height 1.00")
    expect(region(1).style.width).toBe("30%")
  })

  it("does nothing while disabled", () => {
    const store = setup({ disabled: true })
    const box = screen.getByTestId("region-canvas").querySelector<HTMLElement>('[data-region="1"]')!
    expect(box).toHaveAttribute("tabindex", "-1")
    fireEvent.keyDown(box, { key: "ArrowRight" })
    expect(values(store).region1_x).toBe(0)
  })

  it("has a zh-CN translation for every chrome string", () => {
    const sources = import.meta.glob<string>(["./*.tsx", "./*.ts", "!./*.test.*"], { query: "?raw", import: "default", eager: true })
    const keys = new Set([...Object.values(sources).flatMap(widgetKeys), ...LAYOUTS.map((layout) => layout.label)])
    const catalog = zh as Record<string, string>
    expect([...keys].filter((key) => !catalog[key])).toEqual([])
    for (const [key, value] of Object.entries(catalog)) {
      const vars = (text: string) => [...text.matchAll(/\{\{(\w+)\}\}/g)].map((m) => m[1]).sort()
      expect(vars(value), key).toEqual(vars(key))
    }
  })
})
