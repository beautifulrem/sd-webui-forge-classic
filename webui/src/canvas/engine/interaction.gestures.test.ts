/**
 * The gestures of an editor view on a real document (memory rasters): strokes, the eraser
 * (tool or Shift), rectangle and lasso shapes, fill, the eyedropper, panning (hand tool, Space,
 * middle / right button), two-finger pinch, the wheel (zoom, pan, brush adjust) and the cursor.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { View } from "../core/geometry"
import type { Tool } from "../state/tools"
import { EditorDocument } from "./document"
import { type CursorState, Interaction, type InteractionContext } from "./interaction"
import { createMemoryRaster } from "./memory-raster"

const SIZE = 64

function setup(overrides: Partial<InteractionContext> = {}, { image = true }: { image?: boolean } = {}) {
  const doc = new EditorDocument({ createRaster: createMemoryRaster, sync: { encode: async () => new Blob(), upload: async () => "id", delay: 10_000 } })
  if (image) doc.setBackground(null, { width: SIZE, height: SIZE }, "bg")
  const host = document.createElement("div")
  const captured = new Set<number>()
  Object.assign(host, {
    setPointerCapture: (id: number) => captured.add(id),
    releasePointerCapture: (id: number) => captured.delete(id),
    hasPointerCapture: (id: number) => captured.has(id),
  })
  host.getBoundingClientRect = () => ({ left: 10, top: 20, width: SIZE, height: SIZE, right: 10 + SIZE, bottom: 20 + SIZE, x: 10, y: 20, toJSON: () => ({}) })
  const brush = { size: 8, opacity: 0.5, softness: 0 }
  const ctx: InteractionContext = {
    layer: { key: "sketch", kind: "sketch" },
    tool: "brush",
    brush,
    eraser: { size: 16, opacity: 1, softness: 0.5 },
    color: [255, 0, 0],
    fillTolerance: 10,
    fillSample: "layer",
    pressure: false,
    screenBrush: false,
    wheel: "zoom",
    adjust: null,
    ...overrides,
  }
  let view: View = { scale: 1, x: 0, y: 0 }
  const calls = {
    views: [] as View[],
    picked: [] as [number[], boolean][],
    cursors: [] as (CursorState | null)[],
    shapes: [] as [number[] | null, boolean][],
    adjust: [] as [string, number][],
    painted: [] as unknown[],
  }
  const interaction = new Interaction(host, doc, {
    context: () => ctx,
    view: () => view,
    viewport: () => ({ width: SIZE, height: SIZE }),
    setView: (next) => {
      view = next
      calls.views.push(next)
    },
    picked: (color, final) => calls.picked.push([[...color], final]),
    cursor: (state) => calls.cursors.push(state),
    shape: (points, erase) => calls.shapes.push([points ? [...points] : null, erase]),
    adjust: (what, direction) => calls.adjust.push([what, direction]),
    painted: (color) => calls.painted.push(color),
  })
  const setView = (next: View) => {
    view = next
  }
  return { doc, host, interaction, ctx, calls, captured, setView, view: () => view }
}

/** A pointer event at host-local (x, y) (the host sits at 10, 20). */
function pointer(type: string, x: number, y: number, { id = 1, pointerType = "mouse", button = 0, pressure = 0.5 } = {}) {
  return Object.assign(new MouseEvent(type, { clientX: x + 10, clientY: y + 20, button, bubbles: true, cancelable: true }), { pointerId: id, pointerType, pressure })
}

function wheel(x: number, y: number, init: WheelEventInit) {
  return new WheelEvent("wheel", { clientX: x + 10, clientY: y + 20, bubbles: true, cancelable: true, ...init })
}

function drag(host: HTMLElement, points: [number, number][], options: Parameters<typeof pointer>[3] = {}) {
  const [first, ...rest] = points
  host.dispatchEvent(pointer("pointerdown", first![0], first![1], options))
  for (const [x, y] of rest) host.dispatchEvent(pointer("pointermove", x, y, options))
  const last = points[points.length - 1]!
  host.dispatchEvent(pointer("pointerup", last[0], last[1], options))
}

let frames: FrameRequestCallback[] = []

/** Runs the animation frames requested so far. */
function flushFrames() {
  const pending = frames
  frames = []
  for (const callback of pending) callback(0)
}

beforeEach(() => {
  // Frames run when the test says so.
  frames = []
  vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) => frames.push(callback))
  vi.spyOn(window, "cancelAnimationFrame").mockImplementation(() => {})
})

afterEach(() => vi.restoreAllMocks())

describe("strokes", () => {
  it("paints a stroke at the brush opacity and reports the colour", () => {
    const { doc, host, calls, captured, interaction } = setup()
    const begin = vi.spyOn(doc, "beginStroke")
    host.dispatchEvent(pointer("pointerdown", 8, 8))
    expect(captured.has(1)).toBe(true)
    expect(begin).toHaveBeenCalledWith("sketch", "sketch", { size: 8, hardness: 1, opacity: 0.5, color: [255, 0, 0], erase: false, pressure: false })
    host.dispatchEvent(pointer("pointermove", 30, 8))
    host.dispatchEvent(pointer("pointermove", 30, 8, { id: 2 })) // another pointer does not draw into it
    host.dispatchEvent(pointer("pointerup", 30, 8))
    expect(captured.has(1)).toBe(false)
    expect(doc.isEmpty("sketch")).toBe(false)
    expect(calls.painted).toEqual([[255, 0, 0]])
    expect(interaction.busy).toBe(false)
  })

  it("paints masks at full strength whatever the brush opacity", () => {
    const { doc, host } = setup({ layer: { key: "mask", kind: "mask" }, color: null })
    const begin = vi.spyOn(doc, "beginStroke")
    drag(host, [
      [4, 4],
      [20, 4],
    ])
    expect(begin.mock.calls[0]![2]).toMatchObject({ opacity: 1, color: null })
  })

  it("erases with the eraser tool, and with Shift held on the brush, without reporting a colour", () => {
    for (const how of ["tool", "shift"] as const) {
      const { doc, host, interaction, calls } = setup(how === "tool" ? { tool: "eraser" } : {})
      if (how === "shift") interaction.setShift(true)
      const begin = vi.spyOn(doc, "beginStroke")
      drag(host, [
        [4, 4],
        [20, 4],
      ])
      // the eraser's own size and softness
      expect(begin.mock.calls[0]![2]).toMatchObject({ erase: true, size: 16, hardness: 0.5, opacity: 1 })
      expect(calls.painted).toEqual([])
    }
  })

  it("uses pen pressure only with a pen and the pressure option", () => {
    const { doc, host } = setup({ pressure: true })
    const begin = vi.spyOn(doc, "beginStroke")
    const extend = vi.spyOn(doc, "extendStroke")
    drag(
      host,
      [
        [4, 4],
        [12, 4],
      ],
      { pointerType: "pen", pressure: 0.25 },
    )
    expect(begin.mock.calls[0]![2].pressure).toBe(true)
    expect(extend.mock.calls.flatMap(([points]) => points.map((point) => point.pressure))).toEqual([0.25, 0.25])
    drag(host, [
      [4, 4],
      [12, 4],
    ])
    expect(begin.mock.calls[1]![2].pressure).toBe(false)
  })

  it("sizes screen-px brushes by the zoom", () => {
    const { doc, host, setView } = setup({ screenBrush: true })
    setView({ scale: 2, x: 0, y: 0 })
    const begin = vi.spyOn(doc, "beginStroke")
    drag(host, [
      [4, 4],
      [12, 4],
    ])
    expect(begin.mock.calls[0]![2].size).toBe(4)
  })

  it("applies the queued points once per animation frame", () => {
    const { doc, host } = setup()
    const extend = vi.spyOn(doc, "extendStroke")
    host.dispatchEvent(pointer("pointerdown", 4, 4))
    host.dispatchEvent(pointer("pointermove", 8, 4))
    host.dispatchEvent(pointer("pointermove", 12, 4))
    expect(frames).toHaveLength(1) // one frame for all of them
    flushFrames()
    expect(extend).toHaveBeenCalledTimes(1)
    expect(extend.mock.calls[0]![0]).toHaveLength(3)
    host.dispatchEvent(pointer("pointerup", 12, 4))
  })

  it("a cancelled pointer drops the stroke", () => {
    const { doc, host, calls } = setup()
    host.dispatchEvent(pointer("pointerdown", 4, 4))
    host.dispatchEvent(pointer("pointermove", 20, 4))
    host.dispatchEvent(pointer("pointercancel", 20, 4, { id: 2 })) // not its pointer
    expect(doc.stroking).toBe(true)
    host.dispatchEvent(pointer("pointercancel", 20, 4))
    expect(doc.stroking).toBe(false)
    expect(doc.isEmpty("sketch")).toBe(true)
    expect(calls.painted).toEqual([])
  })

  it("ignores other buttons and an image-less document", () => {
    const { doc, host } = setup()
    const begin = vi.spyOn(doc, "beginStroke")
    host.dispatchEvent(pointer("pointerdown", 4, 4, { button: 3 }))
    expect(begin).not.toHaveBeenCalled()
    const empty = setup({}, { image: false })
    const beginEmpty = vi.spyOn(empty.doc, "beginStroke")
    empty.host.dispatchEvent(pointer("pointerdown", 4, 4))
    empty.host.dispatchEvent(wheel(4, 4, { deltaY: -100 }))
    expect(beginEmpty).not.toHaveBeenCalled()
    expect(empty.calls.views).toEqual([])
  })
})

describe("shapes and fill", () => {
  it("fills a dragged rectangle, previewing it while dragging", () => {
    const { doc, host, calls } = setup({ tool: "rect" })
    const polygon = vi.spyOn(doc, "polygon")
    drag(host, [
      [4, 4],
      [10, 10],
      [20, 30],
    ])
    expect(calls.shapes.at(-2)).toEqual([[4, 4, 20, 4, 20, 30, 4, 30], false])
    expect(calls.shapes.at(-1)).toEqual([null, false]) // the preview ends
    expect(polygon).toHaveBeenCalledWith("sketch", "sketch", [4, 4, 20, 4, 20, 30, 4, 30], { color: [255, 0, 0], opacity: 0.5, erase: false })
    expect(calls.painted).toEqual([[255, 0, 0]])
  })

  it("subtracts a shape with Shift and skips shapes without area", () => {
    const { doc, host, interaction, calls } = setup({ tool: "rect" })
    const polygon = vi.spyOn(doc, "polygon")
    drag(host, [
      [4, 4],
      [4, 30],
    ]) // a line: no area
    expect(polygon).not.toHaveBeenCalled()
    interaction.setShift(true)
    drag(host, [
      [4, 4],
      [20, 20],
    ])
    expect(polygon.mock.calls[0]![3]).toMatchObject({ erase: true })
    expect(calls.painted).toEqual([])
  })

  it("draws a lasso from points at least 2 screen px apart", () => {
    const { doc, host, calls } = setup({ tool: "lasso", layer: { key: "mask", kind: "mask" } })
    const polygon = vi.spyOn(doc, "polygon")
    drag(host, [
      [4, 4],
      [5, 4], // too close: skipped
      [30, 4],
      [30, 30],
      [4, 30],
    ])
    expect(calls.shapes.filter(([points]) => points).at(-1)![0]).toEqual([4, 4, 30, 4, 30, 30, 4, 30])
    expect(polygon).toHaveBeenCalledWith("mask", "mask", [4, 4, 30, 4, 30, 30, 4, 30], { color: [255, 0, 0], opacity: 1, erase: false })
  })

  it("a second finger cancels a shape in progress", () => {
    const { host, calls, interaction } = setup({ tool: "lasso" })
    host.dispatchEvent(pointer("pointerdown", 4, 4, { pointerType: "touch" }))
    host.dispatchEvent(pointer("pointermove", 30, 4, { pointerType: "touch" }))
    host.dispatchEvent(pointer("pointerdown", 40, 40, { id: 2, pointerType: "touch" }))
    expect(calls.shapes.at(-1)).toEqual([null, false])
    expect(interaction.busy).toBe(true) // now a pinch
  })

  it("flood-fills at the pointer and reports the colour when something was filled", () => {
    const { doc, host, calls, captured } = setup({ tool: "fill" })
    const fill = vi.spyOn(doc, "fill").mockReturnValueOnce(true).mockReturnValueOnce(false)
    host.dispatchEvent(pointer("pointerdown", 12, 14))
    expect(fill).toHaveBeenCalledWith("sketch", "sketch", 12, 14, 10, "layer", { color: [255, 0, 0], opacity: 0.5, erase: false })
    expect(captured.size).toBe(0) // released at once
    host.dispatchEvent(pointer("pointerdown", 12, 14))
    expect(calls.painted).toEqual([[255, 0, 0]])
  })
})

describe("eyedropper", () => {
  it("samples while dragging and reports the final colour on release", () => {
    const { doc, host, calls } = setup({ tool: "eyedropper" })
    const sample = vi.spyOn(doc, "sample").mockReturnValue([1, 2, 3])
    drag(host, [
      [4, 4],
      [8, 8],
    ])
    expect(sample.mock.calls).toEqual([
      [4, 4, "sketch"],
      [8, 8, "sketch"],
      [8, 8, "sketch"],
    ])
    expect(calls.picked).toEqual([
      [[1, 2, 3], false],
      [[1, 2, 3], false],
      [[1, 2, 3], true],
    ])
  })

  it("samples the composite image on a mask, and reports nothing outside the image", () => {
    const { doc, host, calls } = setup({ tool: "eyedropper", layer: { key: "mask", kind: "mask" } })
    const sample = vi.spyOn(doc, "sample").mockReturnValue(null)
    drag(host, [[4, 4]])
    expect(sample.mock.calls[0]![2]).toBeNull()
    expect(calls.picked).toEqual([])
  })
})

describe("view", () => {
  it.each<[string, Partial<InteractionContext>, { button?: number; space?: boolean; grabbing?: boolean }]>([
    ["the hand tool", { tool: "hand" as Tool }, { grabbing: true }],
    ["Space", {}, { space: true, grabbing: true }],
    ["the middle button", {}, { button: 1 }],
    ["the right button", {}, { button: 2 }],
    ["a view-only editor", { layer: null }, { grabbing: true }],
  ])("pans with %s", (_name, overrides, { button = 0, space = false, grabbing = false }) => {
    const { host, interaction, calls, view } = setup(overrides)
    if (space) interaction.setSpace(true)
    host.dispatchEvent(pointer("pointerdown", 10, 10, { button }))
    if (grabbing) expect(calls.cursors.at(-1)).toMatchObject({ mode: "grabbing" })
    host.dispatchEvent(pointer("pointermove", 13, 14, { button }))
    host.dispatchEvent(pointer("pointermove", 15, 15, { id: 2 })) // another pointer does not pan
    host.dispatchEvent(pointer("pointerup", 13, 14, { button }))
    expect(calls.views).toHaveLength(1)
    expect(view()).toMatchObject({ scale: 1 })
    expect(interaction.busy).toBe(false)
  })

  it("pinches with two fingers; lifting both ends it", () => {
    const { host, interaction, calls, view } = setup({ layer: { key: "mask", kind: "mask" } })
    host.dispatchEvent(pointer("pointerdown", 20, 20, { id: 1, pointerType: "touch" }))
    host.dispatchEvent(pointer("pointerdown", 40, 20, { id: 2, pointerType: "touch" }))
    host.dispatchEvent(pointer("pointermove", 50, 20, { id: 2, pointerType: "touch" }))
    expect(calls.views.length).toBeGreaterThan(0)
    expect(view().scale).toBeGreaterThan(1)
    host.dispatchEvent(pointer("pointerdown", 30, 30, { id: 3, pointerType: "touch" })) // a third finger is ignored
    host.dispatchEvent(pointer("pointerup", 50, 20, { id: 2, pointerType: "touch" }))
    host.dispatchEvent(pointer("pointercancel", 30, 30, { id: 3, pointerType: "touch" }))
    expect(interaction.busy).toBe(true)
    host.dispatchEvent(pointer("pointerup", 20, 20, { id: 1, pointerType: "touch" }))
    expect(interaction.busy).toBe(false)
  })

  it("a pinch cancelled by the browser ends when no finger is left", () => {
    const { host, interaction } = setup()
    host.dispatchEvent(pointer("pointerdown", 20, 20, { id: 1, pointerType: "touch" }))
    host.dispatchEvent(pointer("pointerdown", 40, 20, { id: 2, pointerType: "touch" }))
    host.dispatchEvent(pointer("pointercancel", 20, 20, { id: 1, pointerType: "touch" }))
    expect(interaction.busy).toBe(true)
    host.dispatchEvent(pointer("pointercancel", 40, 20, { id: 2, pointerType: "touch" }))
    expect(interaction.busy).toBe(false)
  })

  it("zooms at the cursor with the wheel, Ctrl+wheel and trackpad pinch", () => {
    const { host, view } = setup()
    host.dispatchEvent(wheel(32, 32, { deltaY: -100 }))
    expect(view().scale).toBeGreaterThan(1)
    const zoomed = view().scale
    host.dispatchEvent(wheel(32, 32, { deltaY: 50, ctrlKey: true }))
    expect(view().scale).toBeLessThan(zoomed)
  })

  it("pans with the wheel by preference, with Shift and with trackpad scrolling", () => {
    const pan = setup({ wheel: "pan" })
    pan.setView({ scale: 4, x: -64, y: -64 })
    pan.host.dispatchEvent(wheel(10, 10, { deltaY: 5, deltaMode: 1 })) // lines: 16 px each
    expect(pan.view()).toEqual({ scale: 4, x: -64, y: -144 })

    const shift = setup()
    shift.setView({ scale: 4, x: -64, y: -64 })
    shift.host.dispatchEvent(wheel(10, 10, { deltaY: 10, shiftKey: true })) // vertical wheel scrolls sideways
    expect(shift.view()).toEqual({ scale: 4, x: -74, y: -64 })

    const trackpad = setup()
    trackpad.setView({ scale: 4, x: -64, y: -64 })
    trackpad.host.dispatchEvent(wheel(10, 10, { deltaX: 6, deltaY: 4 }))
    expect(trackpad.view()).toEqual({ scale: 4, x: -70, y: -68 })

    const pages = setup({ wheel: "pan" })
    pages.setView({ scale: 4, x: -64, y: -64 })
    pages.host.dispatchEvent(wheel(10, 10, { deltaY: -0.1, deltaMode: 2 })) // pages: 400 px each
    expect(pages.view().y).toBeCloseTo(-24)
  })

  it("adjusts the brush with the wheel while W / A / S is held", () => {
    const { host, calls } = setup({ adjust: "size" })
    host.dispatchEvent(wheel(10, 10, { deltaY: -1 }))
    host.dispatchEvent(wheel(10, 10, { deltaY: 1 }))
    host.dispatchEvent(wheel(10, 10, { deltaX: 1 })) // no vertical movement: a pan
    expect(calls.adjust).toEqual([
      ["size", 1],
      ["size", -1],
    ])
    expect(calls.views).toHaveLength(1)
  })
})

describe("cursor", () => {
  it("shows the brush size, the tool and whether the pointer is over the image", () => {
    const { host, interaction, calls, ctx, setView } = setup()
    host.dispatchEvent(pointer("pointermove", 8, 8))
    flushFrames()
    expect(calls.cursors.at(-1)).toEqual({ x: 8, y: 8, diameter: 8, mode: "brush", inside: true })
    setView({ scale: 2, x: 0, y: 0 })
    host.dispatchEvent(pointer("pointermove", 200, 8))
    flushFrames()
    expect(calls.cursors.at(-1)).toEqual({ x: 200, y: 8, diameter: 16, mode: "brush", inside: false })
    interaction.setShift(true)
    expect(calls.cursors.at(-1)).toMatchObject({ mode: "eraser", diameter: 32 })
    interaction.setShift(true) // unchanged: no update
    const count = calls.cursors.length
    interaction.setShift(true)
    expect(calls.cursors).toHaveLength(count)
    interaction.setShift(false)
    interaction.setSpace(true)
    expect(calls.cursors.at(-1)).toMatchObject({ mode: "grab" })
    interaction.setSpace(false)
    for (const [tool, mode] of [
      ["eyedropper", "pick"],
      ["fill", "cross"],
      ["rect", "cross"],
      ["hand", "grab"],
    ] as const) {
      ctx.tool = tool
      host.dispatchEvent(pointer("pointermove", 8, 8))
      flushFrames()
      expect(calls.cursors.at(-1)!.mode).toBe(mode)
    }
    ctx.layer = null
    ctx.tool = "brush"
    host.dispatchEvent(pointer("pointermove", 8, 8))
    flushFrames()
    expect(calls.cursors.at(-1)!.mode).toBe("default") // view only: the plain arrow (dragging still pans)
  })

  it("hides the cursor when the pointer leaves, except during its own drag", () => {
    const { host, calls } = setup()
    host.dispatchEvent(pointer("pointerdown", 8, 8))
    host.dispatchEvent(pointer("pointerleave", 80, 8))
    expect(calls.cursors.at(-1)).not.toBeNull()
    host.dispatchEvent(pointer("pointerup", 80, 8))
    host.dispatchEvent(pointer("pointerleave", 80, 8))
    expect(calls.cursors.at(-1)).toBeNull()
  })

  it("is the default arrow without an image", () => {
    const { host, interaction, calls } = setup({}, { image: false })
    host.dispatchEvent(pointer("pointermove", 8, 8))
    flushFrames()
    interaction.setSpace(true)
    expect(calls.cursors.at(-1)).toMatchObject({ mode: "default" })
  })

  it("prevents the context menu and Safari's page pinch", () => {
    const { host } = setup()
    for (const type of ["contextmenu", "gesturestart"]) {
      const event = new Event(type, { cancelable: true })
      host.dispatchEvent(event)
      expect(event.defaultPrevented).toBe(true)
    }
  })
})

describe("destroy", () => {
  it("commits an erase stroke in progress without reporting a colour, and detaches", () => {
    const { doc, host, interaction, calls } = setup({ tool: "eraser" })
    host.dispatchEvent(pointer("pointerdown", 4, 4))
    host.dispatchEvent(pointer("pointermove", 20, 4))
    interaction.destroy()
    expect(doc.stroking).toBe(false)
    expect(calls.painted).toEqual([])
    const begin = vi.spyOn(doc, "beginStroke")
    host.dispatchEvent(pointer("pointerdown", 4, 4))
    expect(begin).not.toHaveBeenCalled()
  })
})
