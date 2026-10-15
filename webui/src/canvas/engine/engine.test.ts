import { afterEach, describe, expect, it, vi } from "vitest"
import { EditorDocument } from "./document"
import { createMemoryRaster, MemoryRaster } from "./memory-raster"
import { createUploadCache, LayerSync } from "./sync"

const W = 16
const H = 12

function makeDoc(upload = vi.fn(async () => "id")) {
  const doc = new EditorDocument({
    createRaster: createMemoryRaster,
    // Content-dependent "PNGs", so the upload cache tells contents apart.
    sync: { encode: async (data: Uint8ClampedArray) => new Blob([data.slice()]), upload, delay: 10 },
  })
  doc.setBackground(null, { width: W, height: H }, "bg")
  return doc
}

const raster = (doc: EditorDocument, key: string) => doc.layer(key)!.raster as MemoryRaster
const alphaAt = (doc: EditorDocument, key: string, x: number, y: number) => raster(doc, key).data[(y * W + x) * 4 + 3]

function paintLine(doc: EditorDocument, key: string, kind: "mask" | "sketch", options: Partial<Parameters<EditorDocument["beginStroke"]>[2]> = {}) {
  doc.beginStroke(key, kind, { size: 4, hardness: 1, opacity: 1, color: kind === "mask" ? null : [255, 0, 0], erase: false, pressure: false, ...options })
  doc.extendStroke([{ x: 2, y: 6, pressure: 1 }])
  doc.extendStroke([
    { x: 8, y: 6, pressure: 1 },
    { x: 13, y: 6, pressure: 1 },
  ])
  doc.endStroke()
}

afterEach(() => vi.useRealTimers())

describe("EditorDocument", () => {
  it("paints strokes into the layer, undoes and redoes them exactly", () => {
    const doc = makeDoc()
    paintLine(doc, "mask", "mask")
    const painted = raster(doc, "mask").data.slice()
    expect(alphaAt(doc, "mask", 8, 6)).toBe(255)
    expect(alphaAt(doc, "mask", 8, 0)).toBe(0)
    expect(doc.canUndo()).toBe(true)

    expect(doc.undo()).toBe(true)
    expect(raster(doc, "mask").data.every((v, i) => i % 4 !== 3 || v === 0)).toBe(true)
    expect(doc.redo()).toBe(true)
    expect(raster(doc, "mask").data).toEqual(painted)
    expect(doc.redo()).toBe(false)
  })

  it("keeps a stroke out of the layer until it ends, and composites it at the brush opacity once", () => {
    const doc = makeDoc()
    doc.beginStroke("sketch", "sketch", { size: 6, hardness: 1, opacity: 0.5, color: [0, 0, 255], erase: false, pressure: false })
    // Back and forth over the same pixels: dabs overlap but must not build up.
    doc.extendStroke([
      { x: 3, y: 3, pressure: 1 },
      { x: 12, y: 3, pressure: 1 },
      { x: 3, y: 3, pressure: 1 },
    ])
    expect(doc.preview?.opacity).toBe(0.5)
    expect(alphaAt(doc, "sketch", 6, 3)).toBe(0)
    doc.endStroke()
    expect(doc.preview).toBeNull()
    expect(alphaAt(doc, "sketch", 6, 3)).toBe(128)
    expect([...raster(doc, "sketch").data.slice((3 * W + 6) * 4, (3 * W + 6) * 4 + 3)]).toEqual([0, 0, 255])
    // The scratch raster is clean again.
    expect((doc.scratchRaster as MemoryRaster).data.every((v) => v === 0)).toBe(true)
  })

  it("erases with the eraser and restores on undo", () => {
    const doc = makeDoc()
    doc.polygon("mask", "mask", [0, 0, W, 0, W, H, 0, H], { color: null, opacity: 1, erase: false })
    expect(alphaAt(doc, "mask", 1, 1)).toBe(255)
    paintLine(doc, "mask", "mask", { erase: true })
    expect(alphaAt(doc, "mask", 8, 6)).toBe(0)
    expect(alphaAt(doc, "mask", 8, 0)).toBe(255)
    doc.undo()
    expect(alphaAt(doc, "mask", 8, 6)).toBe(255)
  })

  it("exports masks as white + alpha, null when empty", () => {
    const doc = makeDoc()
    doc.ensureLayer("mask", "mask")
    expect(doc.exportLayer("mask")).toBeNull()
    doc.polygon("mask", "mask", [4, 4, 8, 4, 8, 8, 4, 8], { color: null, opacity: 1, erase: false })
    const out = doc.exportLayer("mask")!
    expect(out.width).toBe(W)
    const at = (x: number, y: number) => [...out.data.slice((y * W + x) * 4, (y * W + x) * 4 + 4)]
    expect(at(5, 5)).toEqual([255, 255, 255, 255])
    expect(at(0, 0)).toEqual([255, 255, 255, 0])
    // The display tint is untouched.
    expect(raster(doc, "mask").data.slice((5 * W + 5) * 4, (5 * W + 5) * 4 + 3)).toEqual(new Uint8ClampedArray(doc.tint))
  })

  it("inverts and clears masks undoably", () => {
    const doc = makeDoc()
    doc.polygon("mask", "mask", [0, 0, 4, 0, 4, 4, 0, 4], { color: null, opacity: 1, erase: false })
    doc.invert("mask")
    expect(alphaAt(doc, "mask", 1, 1)).toBe(0)
    expect(alphaAt(doc, "mask", 10, 10)).toBe(255)
    doc.clear("mask")
    expect(doc.isEmpty("mask")).toBe(true)
    doc.undo()
    expect(alphaAt(doc, "mask", 10, 10)).toBe(255)
    doc.undo()
    expect(alphaAt(doc, "mask", 1, 1)).toBe(255)
    expect(alphaAt(doc, "mask", 10, 10)).toBe(0)
    doc.redo()
    expect(alphaAt(doc, "mask", 10, 10)).toBe(255)
  })

  it("fills enclosed areas", () => {
    const doc = makeDoc()
    // A box outline, then a fill inside it.
    doc.polygon("sketch", "sketch", [2, 2, 10, 2, 10, 3, 2, 3], { color: [0, 0, 0], opacity: 1, erase: false })
    doc.polygon("sketch", "sketch", [2, 8, 10, 8, 10, 9, 2, 9], { color: [0, 0, 0], opacity: 1, erase: false })
    doc.polygon("sketch", "sketch", [2, 2, 3, 2, 3, 9, 2, 9], { color: [0, 0, 0], opacity: 1, erase: false })
    doc.polygon("sketch", "sketch", [9, 2, 10, 2, 10, 9, 9, 9], { color: [0, 0, 0], opacity: 1, erase: false })
    expect(doc.fill("sketch", "sketch", 5, 5, 0, "layer", { color: [0, 255, 0], opacity: 1, erase: false })).toBe(true)
    expect([...raster(doc, "sketch").data.slice((5 * W + 5) * 4, (5 * W + 5) * 4 + 4)]).toEqual([0, 255, 0, 255])
    expect(alphaAt(doc, "sketch", 12, 5)).toBe(0)
    doc.undo()
    expect(alphaAt(doc, "sketch", 5, 5)).toBe(0)
    doc.redo()
    expect(alphaAt(doc, "sketch", 5, 5)).toBe(255)
  })

  it("drops layers and history for a new picture", () => {
    const doc = makeDoc()
    paintLine(doc, "mask", "mask")
    doc.setBackground(null, { width: W, height: H }, "other")
    expect(doc.isEmpty("mask")).toBe(true)
    expect(doc.canUndo()).toBe(false)
    doc.setBackground(null, { width: 8, height: 8 }, "small")
    expect(doc.layer("mask")).toBeNull()
  })

  it("forgets the history of a layer replaced from outside, keeping the others'", () => {
    const doc = makeDoc()
    paintLine(doc, "sketch", "sketch")
    paintLine(doc, "mask", "mask")
    doc.resetLayer("mask")
    expect(doc.historyDepth).toBe(1)
    expect(doc.isEmpty("mask")).toBe(true)
    doc.undo()
    expect(doc.isEmpty("sketch")).toBe(true)
  })

  it("undoes and redoes the given layer only, never a hidden one", () => {
    const doc = makeDoc()
    // Sketch mode first, then inpaint: Ctrl+Z in inpaint must not touch the sketch.
    paintLine(doc, "sketch", "sketch")
    doc.polygon("mask", "mask", [0, 0, 4, 0, 4, 4, 0, 4], { color: null, opacity: 1, erase: false })
    paintLine(doc, "sketch", "sketch", { erase: true })
    expect(doc.canUndo("mask")).toBe(true)
    expect(doc.undo("mask")).toBe(true)
    expect(doc.isEmpty("mask")).toBe(true)
    // The sketch's erase (the newest edit overall) is still there.
    expect(alphaAt(doc, "sketch", 8, 6)).toBe(0)
    expect(doc.canUndo("mask")).toBe(false)
    expect(doc.undo("mask")).toBe(false)
    expect(doc.canRedo("mask")).toBe(true)
    expect(doc.canRedo("sketch")).toBe(false)

    // A new sketch edit keeps the mask's redo.
    doc.polygon("sketch", "sketch", [12, 0, 15, 0, 15, 3, 12, 3], { color: [0, 0, 0], opacity: 1, erase: false })
    expect(doc.redo("mask")).toBe(true)
    expect(alphaAt(doc, "mask", 1, 1)).toBe(255)
    expect(doc.undo("sketch")).toBe(true)
    expect(doc.undo("sketch")).toBe(true)
    expect(alphaAt(doc, "sketch", 8, 6)).toBe(255)
    expect(alphaAt(doc, "mask", 1, 1)).toBe(255)
  })

  it("tracks the painted bounds without reading the whole layer", () => {
    const doc = makeDoc()
    doc.polygon("mask", "mask", [2, 2, 6, 2, 6, 6, 2, 6], { color: null, opacity: 1, erase: false })
    const read = vi.spyOn(raster(doc, "mask"), "read")
    const first = doc.contentBounds("mask")
    expect(read).not.toHaveBeenCalled()
    expect(first.x).toBeLessThanOrEqual(2)
    expect(first.x + first.width).toBeGreaterThanOrEqual(6)
    paintLine(doc, "mask", "mask")
    read.mockClear()
    const both = doc.contentBounds("mask")
    expect(read).not.toHaveBeenCalled()
    expect(both.x + both.width).toBeGreaterThanOrEqual(13)
    // Erasing makes the bounds an upper bound: only that rectangle is scanned, and they shrink.
    doc.polygon("mask", "mask", [0, 0, W, 0, W, 5, 0, 5], { color: null, opacity: 1, erase: true })
    read.mockClear()
    const after = doc.contentBounds("mask")
    expect(read).toHaveBeenCalledTimes(1)
    const region = read.mock.calls[0]![0]
    expect(region.width * region.height).toBeLessThan(W * H)
    expect(after.y).toBeGreaterThanOrEqual(5)
    doc.clear("mask")
    expect(doc.contentBounds("mask").width).toBe(0)
  })

  it("samples the colour shown", () => {
    const doc = makeDoc()
    doc.polygon("sketch", "sketch", [0, 0, 4, 0, 4, 4, 0, 4], { color: [10, 20, 30], opacity: 1, erase: false })
    expect(doc.sample(1, 1, "sketch")).toEqual([10, 20, 30])
    expect(doc.sample(10, 10, "sketch")).toEqual([255, 255, 255])
    expect(doc.sample(-1, 0, "sketch")).toBeNull()
  })

  it("uploads a layer after edits and commits its id", async () => {
    vi.useFakeTimers()
    const upload = vi.fn(async () => "up-1")
    const doc = makeDoc(upload)
    const commits: [string, string | null][] = []
    doc.onCommit = (key, id) => commits.push([key, id])
    paintLine(doc, "mask", "mask")
    expect(doc.pending).toBe(true)
    await vi.advanceTimersByTimeAsync(50)
    expect(upload).toHaveBeenCalledTimes(1)
    expect(commits).toEqual([["mask", "up-1"]])
    expect(doc.pending).toBe(false)
    // Undo to empty: null without an upload; redo: the cached id without an upload.
    doc.undo()
    await doc.flush()
    doc.redo()
    await doc.flush()
    expect(upload).toHaveBeenCalledTimes(1)
    expect(commits).toEqual([
      ["mask", "up-1"],
      ["mask", null],
      ["mask", "up-1"],
    ])
  })
})

describe("LayerSync", () => {
  const deps = (read: () => Uint8ClampedArray | null, upload: (blob: Blob) => Promise<string>) => ({
    // Takes the content when called, like canvas.toBlob.
    png: () => {
      const data = read()
      return Promise.resolve(data ? new Blob([data.slice()]) : null)
    },
    upload,
    cache: createUploadCache(),
    delay: 100,
  })

  it("debounces edits and flushes on demand", async () => {
    vi.useFakeTimers()
    let content = 1
    const upload = vi.fn(async () => `id-${content}`)
    const sync = new LayerSync(deps(() => new Uint8ClampedArray([content, 0, 0, 255]), upload))
    const commit = vi.fn()
    sync.bind(commit)
    sync.touch()
    content = 2
    sync.touch()
    await vi.advanceTimersByTimeAsync(50)
    expect(upload).not.toHaveBeenCalled()
    await sync.flush()
    expect(upload).toHaveBeenCalledTimes(1)
    expect(commit).toHaveBeenLastCalledWith("id-2")
    expect(sync.pending).toBe(false)
  })

  it("re-exports edits made while an upload is in flight", async () => {
    let content = 1
    let release: () => void = () => {}
    const upload = vi.fn(
      (): Promise<string> =>
        new Promise((resolve) => {
          const value = `id-${content}`
          release = () => resolve(value)
        }),
    )
    const sync = new LayerSync(deps(() => new Uint8ClampedArray([content, 0, 0, 255]), upload))
    const commit = vi.fn()
    sync.bind(commit)
    sync.touch()
    const first = sync.flush()
    await vi.waitFor(() => expect(upload).toHaveBeenCalledTimes(1))
    content = 2
    sync.touch()
    const second = sync.flush()
    release()
    await vi.waitFor(() => expect(upload).toHaveBeenCalledTimes(2))
    release()
    await Promise.all([first, second])
    expect(commit).toHaveBeenLastCalledWith("id-2")
    expect(sync.value).toBe("id-2")
  })

  it("adopts values set from outside without uploading", async () => {
    const upload = vi.fn(async () => "x")
    const sync = new LayerSync(deps(() => null, upload))
    const commit = vi.fn()
    sync.bind(commit)
    sync.touch()
    sync.adopt("external")
    await sync.flush()
    expect(upload).not.toHaveBeenCalled()
    expect(commit).not.toHaveBeenCalled()
    expect(sync.value).toBe("external")
  })
})

describe("upload cache", () => {
  it("reuses an upload for the same content, but not once the server may have pruned it", () => {
    let now = 0
    const cache = createUploadCache(4, 1000, () => now)
    cache.set("h", "id-1")
    now = 999
    expect(cache.get("h")).toBe("id-1")
    now = 2000
    expect(cache.get("h")).toBeUndefined()
    // Gone for good: a fresh upload replaces it.
    now = 2001
    expect(cache.get("h")).toBeUndefined()
  })
})

describe("undo budget", () => {
  it("scales with the device memory, within 96–384 MiB", async () => {
    const { historyBudget } = await import("./registry")
    const MiB = 1024 * 1024
    expect(historyBudget(0.5, false)).toBe(96 * MiB)
    expect(historyBudget(4, false)).toBe(128 * MiB)
    expect(historyBudget(8, false)).toBe(256 * MiB)
    expect(historyBudget(64, false)).toBe(384 * MiB)
  })

  it("is conservative when the memory is unknown, and lower on touch devices", async () => {
    const { historyBudget } = await import("./registry")
    const MiB = 1024 * 1024
    // Safari / Firefox do not report it: less than an 8 GB Chrome gets, never the maximum.
    expect(historyBudget(undefined, false)).toBe(160 * MiB)
    expect(historyBudget(undefined, false)).toBeLessThan(historyBudget(8, false))
    expect(historyBudget(undefined, true)).toBe(96 * MiB)
    expect(historyBudget(8, true)).toBe(96 * MiB)
  })
})
