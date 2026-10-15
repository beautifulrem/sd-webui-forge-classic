/** Input handling that must not lose the user's paint, and the painting flag the stage reads. */
import { afterEach, describe, expect, it, vi } from "vitest"
import { isPainting } from "../state/activity"
import { EditorDocument } from "./document"
import { Interaction, type InteractionContext } from "./interaction"
import { createMemoryRaster } from "./memory-raster"

function setup(layer: InteractionContext["layer"] = { key: "mask", kind: "mask" }) {
  const doc = new EditorDocument({ createRaster: createMemoryRaster, sync: { encode: async () => new Blob(), upload: async () => "id", delay: 10_000 } })
  doc.setBackground(null, { width: 32, height: 32 }, "bg")
  const host = document.createElement("div")
  Object.assign(host, { setPointerCapture: () => {}, releasePointerCapture: () => {}, hasPointerCapture: () => false })
  host.getBoundingClientRect = () => ({ left: 0, top: 0, width: 32, height: 32, right: 32, bottom: 32, x: 0, y: 0, toJSON: () => ({}) })
  const brush = { size: 6, opacity: 1, softness: 0 }
  const context = (): InteractionContext => ({
    layer,
    tool: "brush",
    brush,
    eraser: brush,
    color: null,
    fillTolerance: 0,
    fillSample: "layer",
    pressure: false,
    screenBrush: false,
    wheel: "zoom",
    adjust: null,
  })
  const interaction = new Interaction(host, doc, {
    context,
    view: () => ({ scale: 1, x: 0, y: 0 }),
    viewport: () => ({ width: 32, height: 32 }),
    setView: () => {},
    picked: () => {},
    cursor: () => {},
    shape: () => {},
    adjust: () => {},
    painted: () => {},
  })
  return { doc, host, interaction }
}

const pointer = (type: string, x: number, y: number, pointerType = "mouse") =>
  Object.assign(new MouseEvent(type, { clientX: x, clientY: y, button: 0, bubbles: true, cancelable: true }), { pointerId: 1, pointerType, pressure: 0.5 })

afterEach(() => vi.useRealTimers())

describe("Interaction", () => {
  it("commits a stroke in progress when the editor goes away (never throws the paint away)", () => {
    const { doc, host, interaction } = setup()
    host.dispatchEvent(pointer("pointerdown", 4, 16))
    host.dispatchEvent(pointer("pointermove", 20, 16))
    expect(doc.stroking).toBe(true)
    expect(isPainting()).toBe(true)
    interaction.destroy()
    expect(doc.stroking).toBe(false)
    expect(isPainting()).toBe(false)
    expect(doc.isEmpty("mask")).toBe(false)
    expect(doc.canUndo("mask")).toBe(true)
  })

  it("leaves a one-finger touch to the page when there is nothing to paint", () => {
    const { doc, host, interaction } = setup(null)
    const down = pointer("pointerdown", 4, 16, "touch")
    host.dispatchEvent(down)
    // Not taken (no preventDefault): the browser scrolls the page.
    expect(down.defaultPrevented).toBe(false)
    expect(interaction.busy).toBe(false)
    expect(doc.stroking).toBe(false)
    interaction.destroy()
  })
})
