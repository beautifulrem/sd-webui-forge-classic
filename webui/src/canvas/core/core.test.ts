import { describe, expect, it } from "vitest"
import { floodFill, regionPixels } from "./flood-fill"
import { clipRect, constrainView, fitView, nextZoomStep, pinchView, rectFromBounds, screenToImage, unionRect, zoomAt } from "./geometry"
import { History } from "./history"
import { aspectFrame, blurReach, cropRegion, expandCropRegion, inpaintArea } from "./overlays"
import { decodePatch, encodePatch, patchBytes } from "./patch"
import { alphaBounds, hashBytes, maskAlphaFromImage, parseHex, tintAlpha, toHex, toMaskExport } from "./pixels"
import { commandFor, isTyping, shortcutList } from "./shortcuts"
import { StrokeWalker, strokeDabs } from "./stroke"
import { sizeToSlider, sliderToSize, stepSize } from "../state/tools"

const entry = (bytes: number, name = "") => ({ bytes, name })

describe("History", () => {
  it("undoes and redoes in order, a push drops the redo branch", () => {
    const history = new History<ReturnType<typeof entry>>()
    history.push(entry(1, "a"))
    history.push(entry(1, "b"))
    expect(history.undo()?.name).toBe("b")
    expect(history.canRedo).toBe(true)
    expect(history.redo()?.name).toBe("b")
    expect(history.undo()?.name).toBe("b")
    history.push(entry(1, "c"))
    expect(history.canRedo).toBe(false)
    expect(history.undo()?.name).toBe("c")
    expect(history.undo()?.name).toBe("a")
    expect(history.undo()).toBeNull()
  })

  it("keeps at least 32 small states by default and caps the count", () => {
    const history = new History<ReturnType<typeof entry>>()
    for (let i = 0; i < 100; i++) history.push(entry(100))
    expect(history.depth).toBeGreaterThanOrEqual(32)
    expect(history.depth).toBe(64)
  })

  it("drops the oldest states past the byte budget but keeps the newest", () => {
    const history = new History<ReturnType<typeof entry>>({ maxEntries: 100, maxBytes: 1000 })
    for (let i = 0; i < 10; i++) history.push(entry(300, String(i)))
    expect(history.depth).toBe(3)
    expect(history.bytes).toBe(900)
    history.push(entry(5000, "huge"))
    expect(history.depth).toBe(1)
    expect(history.undo()?.name).toBe("huge")
    // Redo entries count against the budget until a push drops them.
    expect(history.bytes).toBe(5000)
    history.push(entry(1))
    expect(history.bytes).toBe(1)
  })

  it("removes the entries of one layer", () => {
    const history = new History<{ bytes: number; layer: string }>()
    history.push({ bytes: 1, layer: "mask" })
    history.push({ bytes: 1, layer: "sketch" })
    history.push({ bytes: 1, layer: "mask" })
    history.undo()
    history.removeWhere((e) => e.layer === "mask")
    expect(history.depth).toBe(1)
    expect(history.canRedo).toBe(false)
    expect(history.bytes).toBe(1)
  })
})

describe("patches", () => {
  const pixels = (values: number[][]) => new Uint8ClampedArray(values.flat())

  it("encodes transparent rectangles as empty", () => {
    const patch = encodePatch(new Uint8ClampedArray(4 * 100), 10, 10, "rgba")
    expect(patch.kind).toBe("empty")
    expect(decodePatch(patch)).toEqual(new Uint8ClampedArray(400))
  })

  it("run-length encodes flat colour and round-trips exactly", () => {
    const flat = new Uint8ClampedArray(64 * 64 * 4)
    for (let i = 0; i < flat.length; i += 4) flat.set(i < flat.length / 2 ? [255, 0, 0, 255] : [0, 0, 255, 128], i)
    const patch = encodePatch(flat.slice(), 64, 64, "rgba")
    expect(patch.kind).toBe("rle")
    expect(patchBytes(patch)).toBeLessThan(flat.byteLength / 100)
    expect(decodePatch(patch)).toEqual(flat)
  })

  it("keeps noisy pixels raw", () => {
    const noisy = new Uint8ClampedArray(16 * 16 * 4).map((_, i) => (i * 7919) % 251)
    const patch = encodePatch(noisy.slice(), 16, 16, "rgba")
    expect(patch.kind).toBe("raw")
    expect(decodePatch(patch)).toEqual(noisy)
  })

  it("stores masks as alpha and decodes them in the current tint", () => {
    const mask = pixels([
      [255, 0, 0, 255],
      [255, 0, 0, 0],
      [255, 0, 0, 77],
      [255, 0, 0, 255],
    ])
    const patch = encodePatch(mask, 2, 2, "alpha")
    expect(patch.kind).not.toBe("empty")
    expect([...decodePatch(patch, [1, 2, 3])]).toEqual([1, 2, 3, 255, 1, 2, 3, 0, 1, 2, 3, 77, 1, 2, 3, 255])
  })
})

describe("mask export", () => {
  it("exports white with the painted alpha, whatever the display tint", () => {
    const data = new Uint8ClampedArray([255, 59, 107, 255, 255, 59, 107, 128, 0, 0, 0, 0, 12, 34, 56, 1])
    expect(toMaskExport(data)).toBe(true)
    expect([...data]).toEqual([255, 255, 255, 255, 255, 255, 255, 128, 255, 255, 255, 0, 255, 255, 255, 1])
  })

  it("reports an empty mask", () => {
    expect(toMaskExport(new Uint8ClampedArray(16))).toBe(false)
  })

  it("reads masks like the server: alpha when not opaque, else luminance", () => {
    const alpha = maskAlphaFromImage(new Uint8ClampedArray([0, 0, 0, 200, 255, 255, 255, 0]))
    expect([...alpha]).toEqual([200, 0])
    const luminance = maskAlphaFromImage(new Uint8ClampedArray([255, 255, 255, 255, 0, 0, 0, 255, 255, 0, 0, 255]))
    expect([...luminance]).toEqual([255, 0, 76])
    expect([...tintAlpha(alpha, [1, 2, 3])]).toEqual([1, 2, 3, 200, 1, 2, 3, 0])
  })

  it("finds the painted bounds", () => {
    const data = new Uint8ClampedArray(5 * 4 * 4)
    data[(1 * 5 + 2) * 4 + 3] = 9
    data[(3 * 5 + 4) * 4 + 3] = 255
    expect(alphaBounds(data, 5, 4)).toEqual({ x: 2, y: 1, width: 3, height: 3 })
    expect(alphaBounds(new Uint8ClampedArray(16), 2, 2).width).toBe(0)
  })

  it("hashes content", () => {
    const a = new Uint8ClampedArray([1, 2, 3, 4, 5])
    expect(hashBytes(a)).toBe(hashBytes(new Uint8ClampedArray([1, 2, 3, 4, 5])))
    expect(hashBytes(a)).not.toBe(hashBytes(new Uint8ClampedArray([1, 2, 3, 4, 6])))
  })

  it("parses and prints colours", () => {
    expect(parseHex("#f0a")).toEqual([255, 0, 170])
    expect(parseHex("12ab34")).toEqual([18, 171, 52])
    expect(parseHex("nope")).toBeNull()
    expect(toHex([255, 0, 170])).toBe("#ff00aa")
  })
})

describe("flood fill", () => {
  // 6×4: a wall of opaque pixels (columns 1–3) around a 1×2 transparent hole.
  const W = 6
  const H = 4
  const image = () => {
    const data = new Uint8ClampedArray(W * H * 4)
    const wall = [
      [1, 0], [2, 0], [3, 0],
      [1, 1], [3, 1],
      [1, 2], [3, 2],
      [1, 3], [2, 3], [3, 3],
    ]
    for (const [x, y] of wall) data.set([0, 0, 0, 255], ((y! * W) + x!) * 4)
    return data
  }

  it("fills the enclosed area only", () => {
    const region = floodFill(image(), W, H, 2, 1)
    expect(region.count).toBe(2)
    expect(region.bounds).toEqual({ x: 2, y: 1, width: 1, height: 2 })
    expect([...regionPixels(region, W, [9, 8, 7])]).toEqual([9, 8, 7, 255, 9, 8, 7, 255])
  })

  it("stops at the wall on either side", () => {
    // The wall spans every row, so the left column and the two right columns are separate.
    expect(floodFill(image(), W, H, 0, 0).count).toBe(4)
    expect(floodFill(image(), W, H, 5, 3).count).toBe(8)
  })

  it("respects the tolerance", () => {
    const data = new Uint8ClampedArray([100, 100, 100, 255, 110, 100, 100, 255, 200, 100, 100, 255])
    expect(floodFill(data, 3, 1, 0, 0, 5).count).toBe(1)
    expect(floodFill(data, 3, 1, 0, 0, 10).count).toBe(2)
    expect(floodFill(data, 3, 1, 9, 0).count).toBe(0)
  })
})

describe("strokes", () => {
  const points = [
    { x: 0, y: 0, pressure: 1 },
    { x: 30, y: 0, pressure: 1 },
    { x: 30, y: 17.5, pressure: 0.5 },
    { x: 31, y: 18, pressure: 0.5 },
    { x: 80, y: 40, pressure: 0.2 },
  ]
  const spec = { size: 10, spacing: 0.2, pressure: true }

  it("spaces dabs by the size", () => {
    const dabs = strokeDabs(points.slice(0, 2), { ...spec, pressure: false })
    const xs = dabs.filter((_, i) => i % 3 === 0)
    expect(xs).toEqual([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
  })

  it("gives the same dabs however the points are batched", () => {
    const whole = strokeDabs(points, spec)
    const walker = new StrokeWalker(spec)
    const batched = [...walker.add(points.slice(0, 1)), ...walker.add(points.slice(1, 3)), ...walker.add([]), ...walker.add(points.slice(3))]
    expect(batched.length).toBe(whole.length)
    batched.forEach((value, i) => expect(value).toBeCloseTo(whole[i]!, 9))
  })

  it("scales dabs with pen pressure", () => {
    const dabs = strokeDabs([{ x: 0, y: 0, pressure: 0.5 }], spec)
    expect(dabs[2]).toBe(2.5)
  })
})

describe("geometry", () => {
  it("builds and combines rects", () => {
    expect(rectFromBounds(2.5, 3.2, 0.5, 1)).toEqual({ x: 0, y: 1, width: 3, height: 3 })
    expect(unionRect({ x: 0, y: 0, width: 2, height: 2 }, { x: 5, y: 5, width: 1, height: 1 })).toEqual({ x: 0, y: 0, width: 6, height: 6 })
    expect(clipRect({ x: -5, y: 2, width: 10, height: 100 }, { width: 4, height: 8 })).toEqual({ x: 0, y: 2, width: 4, height: 6 })
  })

  it("fits, zooms at a point and pinches", () => {
    const view = fitView({ width: 1000, height: 500 }, { width: 532, height: 532 })
    expect(view.scale).toBeCloseTo(0.5)
    expect(view).toMatchObject({ x: 16, y: 141 })
    const at = { x: 100, y: 200 }
    const before = screenToImage(view, at)
    const zoomed = zoomAt(view, 2, at)
    expect(screenToImage(zoomed, at).x).toBeCloseTo(before.x)
    expect(screenToImage(zoomed, at).y).toBeCloseTo(before.y)
    const pinched = pinchView(view, [{ x: 0, y: 0 }, { x: 10, y: 0 }], [{ x: 0, y: 0 }, { x: 20, y: 0 }])
    expect(pinched.scale).toBeCloseTo(1)
    expect(nextZoomStep(1, 1)).toBe(1.5)
    expect(nextZoomStep(1, -1)).toBe(0.75)
  })

  it("keeps part of the image in the viewport", () => {
    const view = constrainView({ scale: 1, x: 5000, y: -5000 }, { width: 100, height: 100 }, { width: 400, height: 300 })
    expect(view.x).toBe(352)
    expect(view.y).toBe(-52)
  })
})

describe("overlays", () => {
  const image = { width: 1000, height: 500 }

  it("frames the target ratio per resize mode", () => {
    expect(aspectFrame(image, { width: 512, height: 512 }, 1)).toEqual({ kind: "crop", rect: { x: 250, y: 0, width: 500, height: 500 } })
    expect(aspectFrame(image, { width: 512, height: 512 }, "Resize and Fill")).toEqual({ kind: "fill", rect: { x: 0, y: -250, width: 1000, height: 1000 } })
    expect(aspectFrame(image, { width: 512, height: 512 }, 0).kind).toBe("stretch")
    expect(aspectFrame(image, { width: 512, height: 512 }, 4).kind).toBe("fit")
  })

  it("matches the server's crop region maths", () => {
    expect(blurReach(4)).toBe(10)
    expect(cropRegion({ x: 10, y: 10, width: 20, height: 5 }, 32, image)).toEqual([0, 0, 62, 47])
    // A 128×32 mask processed at 512×512 grows to 128×128 (masking.expand_crop_region's example).
    expect(expandCropRegion([100, 200, 228, 232], { width: 512, height: 512 }, image)).toEqual([100, 152, 228, 280])
    // Pushed back inside the image at the bottom.
    expect(expandCropRegion([100, 480, 228, 500], { width: 512, height: 512 }, image)).toEqual([100, 372, 228, 500])
    const area = inpaintArea({ maskBounds: { x: 100, y: 200, width: 128, height: 32 }, image, target: { width: 512, height: 512 }, blur: 0, padding: 0, inverted: false })
    expect(area).toEqual({ x: 100, y: 152, width: 128, height: 128 })
    expect(inpaintArea({ maskBounds: { x: 0, y: 0, width: 0, height: 0 }, image, target: { width: 1, height: 1 }, blur: 0, padding: 0, inverted: false })).toBeNull()
  })
})

describe("shortcuts", () => {
  const key = (code: string, key: string, mods: Partial<Record<"ctrlKey" | "metaKey" | "altKey" | "shiftKey", boolean>> = {}) => ({
    code,
    key,
    ctrlKey: false,
    metaKey: false,
    altKey: false,
    shiftKey: false,
    ...mods,
  })

  it("maps keys to commands with the platform's modifier", () => {
    expect(commandFor(key("KeyB", "b"))).toEqual({ type: "tool", tool: "brush" })
    expect(commandFor(key("KeyZ", "z", { ctrlKey: true }))).toEqual({ type: "undo" })
    expect(commandFor(key("KeyZ", "z", { ctrlKey: true, shiftKey: true }))).toEqual({ type: "redo" })
    expect(commandFor(key("KeyY", "y", { metaKey: true }), true)).toEqual({ type: "redo" })
    // Cmd on Windows / Ctrl on a Mac is not the modifier.
    expect(commandFor(key("KeyZ", "z", { metaKey: true }))).toBeNull()
    expect(commandFor(key("KeyR", "r"))).toEqual({ type: "fit" })
    expect(commandFor(key("Digit1", "1"))).toEqual({ type: "actual" })
    expect(commandFor(key("KeyF", "f"))).toEqual({ type: "maximize" })
    expect(commandFor(key("BracketRight", "]"))).toEqual({ type: "size", direction: 1 })
    expect(commandFor(key("Slash", "?", { shiftKey: true }))).toEqual({ type: "help" })
    expect(commandFor(key("KeyI", "i", { ctrlKey: true }))).toEqual({ type: "invert" })
    expect(commandFor(key("KeyB", "B", { shiftKey: true }))).toBeNull()
  })

  it("lists every command once per key combination", () => {
    const list = shortcutList(false)
    const combos = list.map((item) => `${item.label}:${item.keys.join("+")}`)
    expect(new Set(combos).size).toBe(combos.length)
    expect(list.find((item) => item.label === "Undo")?.keys).toEqual(["Ctrl", "Z"])
    expect(shortcutList(true).find((item) => item.label === "Undo")?.keys).toEqual(["⌘", "Z"])
  })

  it("stays out of text fields", () => {
    const input = document.createElement("input")
    const range = document.createElement("input")
    range.type = "range"
    expect(isTyping(input)).toBe(true)
    expect(isTyping(range)).toBe(false)
    expect(isTyping(document.createElement("button"))).toBe(false)
    expect(isTyping(document.createElement("textarea"))).toBe(true)
  })
})

describe("brush size scale", () => {
  it("maps the slider to sizes on a log scale and steps by about 12 %", () => {
    expect(sliderToSize(0)).toBe(1)
    expect(sliderToSize(100)).toBe(1024)
    expect(sliderToSize(50)).toBe(32)
    for (const size of [1, 7, 48, 300, 1024]) expect(sliderToSize(sizeToSlider(size))).toBe(size)
    expect(stepSize(48, 1)).toBe(54)
    expect(stepSize(2, -1)).toBe(1)
    expect(stepSize(1024, 1)).toBe(1024)
  })
})
