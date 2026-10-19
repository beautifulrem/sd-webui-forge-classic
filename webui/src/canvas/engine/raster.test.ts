/**
 * CanvasRaster, the browser raster: which 2D-canvas operations each raster operation issues
 * (jsdom has no canvas backend, so a recording context stands in; the pixels themselves are
 * checked on MemoryRaster and in the Playwright canvas tests in Chromium).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { CanvasRaster, createCanvasRaster, encodePng, HARD } from "./raster"

type Call = [string, ...unknown[]]

/** A 2D context that records method calls and property writes, per canvas. */
class RecordingContext {
  calls: Call[] = []
  readonly canvas: HTMLCanvasElement
  private state: Record<string, unknown> = { globalAlpha: 1, globalCompositeOperation: "source-over", fillStyle: "#000000" }
  private saved: Record<string, unknown>[] = []

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
    for (const name of ["globalAlpha", "globalCompositeOperation", "fillStyle"]) {
      Object.defineProperty(this, name, {
        get: () => this.state[name],
        set: (value) => {
          this.calls.push([`${name}=`, value])
          this.state[name] = value
        },
      })
    }
  }

  private record(name: string, args: unknown[]) {
    this.calls.push([name, ...args])
  }

  save() {
    this.saved.push({ ...this.state })
    this.record("save", [])
  }
  restore() {
    this.state = this.saved.pop() ?? this.state
    this.record("restore", [])
  }
  getImageData(x: number, y: number, width: number, height: number) {
    this.record("getImageData", [x, y, width, height])
    return { data: new Uint8ClampedArray(width * height * 4).fill(7) }
  }
  putImageData(image: { width: number; height: number }, x: number, y: number) {
    this.record("putImageData", [image.width, image.height, x, y])
  }
  clearRect(...args: number[]) {
    this.record("clearRect", args)
  }
  fillRect(...args: number[]) {
    this.record("fillRect", args)
  }
  beginPath() {
    this.record("beginPath", [])
  }
  closePath() {
    this.record("closePath", [])
  }
  moveTo(...args: number[]) {
    this.record("moveTo", args)
  }
  lineTo(...args: number[]) {
    this.record("lineTo", args)
  }
  arc(...args: number[]) {
    this.record("arc", args)
  }
  fill(...args: unknown[]) {
    this.record("fill", args)
  }
  drawImage(source: unknown, ...args: number[]) {
    this.record("drawImage", [source, ...args])
  }
  createRadialGradient(...args: number[]) {
    this.record("createRadialGradient", args)
    const stops: [number, string][] = []
    return { stops, addColorStop: (offset: number, color: string) => stops.push([offset, color]) }
  }
}

const contexts = new Map<HTMLCanvasElement, RecordingContext>()
const contextOf = (canvas: HTMLCanvasElement) => contexts.get(canvas)!
let blobResult: Blob | null = new Blob(["png"], { type: "image/png" })

beforeEach(() => {
  contexts.clear()
  blobResult = new Blob(["png"], { type: "image/png" })
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(function (this: HTMLCanvasElement) {
    let context = contexts.get(this)
    if (!context) {
      context = new RecordingContext(this)
      contexts.set(this, context)
    }
    return context as unknown as CanvasRenderingContext2D
  } as unknown as HTMLCanvasElement["getContext"])
  vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation(function (callback: BlobCallback, type?: string) {
    expect(type).toBe("image/png")
    queueMicrotask(() => callback(blobResult))
  })
  if (typeof globalThis.ImageData === "undefined") {
    vi.stubGlobal(
      "ImageData",
      class {
        readonly data: Uint8ClampedArray
        readonly width: number
        readonly height: number
        constructor(data: Uint8ClampedArray, width: number, height: number) {
          this.data = data
          this.width = width
          this.height = height
        }
      },
    )
  }
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

const names = (calls: Call[]) => calls.map((call) => call[0])

describe("CanvasRaster", () => {
  it("is a canvas of the raster's size, and fails clearly without a 2D context", () => {
    const raster = createCanvasRaster(30, 20, { readFrequently: true })
    expect(raster.source).toBeInstanceOf(HTMLCanvasElement)
    expect([raster.width, raster.height, raster.source!.width, raster.source!.height]).toEqual([30, 20, 30, 20])
    expect(HTMLCanvasElement.prototype.getContext).toHaveBeenCalledWith("2d", { willReadFrequently: true })
    vi.mocked(HTMLCanvasElement.prototype.getContext).mockReturnValueOnce(null)
    expect(() => new CanvasRaster(4, 4)).toThrow("2D canvas unavailable")
  })

  it("reads, writes and clears pixels", () => {
    const raster = new CanvasRaster(8, 8)
    const ctx = contextOf(raster.source)
    expect(raster.read({ x: 1, y: 2, width: 2, height: 1 })).toEqual(new Uint8ClampedArray(8).fill(7))
    raster.write({ x: 3, y: 4, width: 1, height: 2 }, new Uint8ClampedArray(8))
    raster.clear({ x: 1, y: 1, width: 2, height: 2 })
    raster.clear()
    expect(ctx.calls).toEqual([
      ["getImageData", 1, 2, 2, 1],
      ["putImageData", 1, 2, 3, 4],
      ["clearRect", 1, 1, 2, 2],
      ["clearRect", 0, 0, 8, 8],
    ])
  })

  it("draws hard dabs as one path of discs, filled once", () => {
    const raster = new CanvasRaster(64, 64)
    const ctx = contextOf(raster.source)
    raster.dabs([10, 12, 3, 20, 12, 4, 99], HARD, [255, 0, 16]) // a trailing partial dab is ignored
    expect(ctx.calls).toEqual([["fillStyle=", "#ff0010"], ["beginPath"], ["moveTo", 13, 12], ["arc", 10, 12, 3, 0, Math.PI * 2], ["moveTo", 24, 12], ["arc", 20, 12, 4, 0, Math.PI * 2], ["fill"]])
    ctx.calls = []
    raster.dabs([1, 2], 1, [0, 0, 0])
    expect(ctx.calls).toEqual([])
  })

  it("stamps soft dabs from a cached gradient texture per hardness and colour", () => {
    const raster = new CanvasRaster(64, 64)
    const ctx = contextOf(raster.source)
    raster.dabs([10, 10, 5, 30, 30, 2], 0.5, [0, 128, 255])
    const draws = ctx.calls.filter((call) => call[0] === "drawImage")
    expect(draws.map((call) => call.slice(2))).toEqual([
      [5, 5, 10, 10],
      [28, 28, 4, 4],
    ])
    const stamp = draws[0]![1] as HTMLCanvasElement
    expect(draws[1]![1]).toBe(stamp)
    expect([stamp.width, stamp.height]).toEqual([256, 256])
    const stampCtx = contextOf(stamp)
    const gradient = stampCtx.calls.find((call) => call[0] === "createRadialGradient")
    expect(gradient).toEqual(["createRadialGradient", 128, 128, 0, 128, 128, 128])
    expect(stampCtx.calls.at(-1)).toEqual(["fillRect", 0, 0, 256, 256])

    // the same hardness and colour: the cached stamp; another one: a new stamp
    raster.dabs([1, 1, 1], 0.5, [0, 128, 255])
    raster.dabs([1, 1, 1], 0.2, [0, 128, 255])
    const stamps = ctx.calls.filter((call) => call[0] === "drawImage").map((call) => call[1])
    expect(stamps[2]).toBe(stamp)
    expect(stamps[3]).not.toBe(stamp)
  })

  it("keeps at most 12 stamp textures", () => {
    const raster = new CanvasRaster(8, 8)
    const ctx = contextOf(raster.source)
    for (let i = 0; i < 14; i++) raster.dabs([1, 1, 1], 0.01 * i, [i, 0, 0])
    const first = ctx.calls.find((call) => call[0] === "drawImage")![1]
    const created = contexts.size
    raster.dabs([1, 1, 1], 0, [0, 0, 0]) // the oldest was evicted: drawn again
    expect(contexts.size).toBe(created + 1)
    expect(ctx.calls.filter((call) => call[0] === "drawImage").at(-1)![1]).not.toBe(first)
  })

  it("fills polygons with the non-zero rule and skips fewer than three points", () => {
    const raster = new CanvasRaster(16, 16)
    const ctx = contextOf(raster.source)
    raster.polygon([0, 0, 4, 0], [0, 0, 0])
    expect(ctx.calls).toEqual([])
    raster.polygon([0, 0, 4, 0, 4, 4, 9], [1, 2, 3])
    expect(ctx.calls).toEqual([["fillStyle=", "#010203"], ["beginPath"], ["moveTo", 0, 0], ["lineTo", 4, 0], ["lineTo", 4, 4], ["closePath"], ["fill", "nonzero"]])
  })

  it("composites a scratch raster at a clamped opacity, painting or erasing, and restores the state", () => {
    const layer = new CanvasRaster(16, 16)
    const scratch = new CanvasRaster(16, 16)
    const ctx = contextOf(layer.source)
    const rect = { x: 2, y: 3, width: 4, height: 5 }
    layer.composite(scratch, rect, 1.5, false)
    layer.composite(scratch, rect, -1, true)
    expect(ctx.calls).toEqual([
      ["save"],
      ["globalAlpha=", 1],
      ["globalCompositeOperation=", "source-over"],
      ["drawImage", scratch.source, 2, 3, 4, 5, 2, 3, 4, 5],
      ["restore"],
      ["save"],
      ["globalAlpha=", 0],
      ["globalCompositeOperation=", "destination-out"],
      ["drawImage", scratch.source, 2, 3, 4, 5, 2, 3, 4, 5],
      ["restore"],
    ])
    expect(() => layer.composite({ ...scratch, source: null } as never, rect, 1, false)).toThrow("composite needs a canvas raster")
  })

  it("tints keeping alpha, and inverts a mask through a scratch canvas it frees", () => {
    const raster = new CanvasRaster(6, 4)
    const ctx = contextOf(raster.source)
    raster.tint([255, 255, 0])
    expect(ctx.calls).toEqual([["save"], ["globalCompositeOperation=", "source-in"], ["fillStyle=", "#ffff00"], ["fillRect", 0, 0, 6, 4], ["restore"]])

    ctx.calls = []
    raster.invert([0, 0, 255])
    const scratch = ctx.calls.find((call) => call[0] === "drawImage")![1] as HTMLCanvasElement
    expect(names(contextOf(scratch).calls)).toEqual(["fillStyle=", "fillRect", "globalCompositeOperation=", "drawImage"])
    expect(contextOf(scratch).calls[2]).toEqual(["globalCompositeOperation=", "destination-out"])
    expect(contextOf(scratch).calls[3]![1]).toBe(raster.source)
    expect(ctx.calls).toEqual([
      ["clearRect", 0, 0, 6, 4],
      ["drawImage", scratch, 0, 0],
    ])
    expect([scratch.width, scratch.height]).toEqual([0, 0])
  })

  it("draws an image over the whole raster and frees its backing store on dispose", () => {
    const raster = new CanvasRaster(6, 4)
    const image = document.createElement("canvas")
    raster.drawImage(image)
    expect(contextOf(raster.source).calls).toEqual([["drawImage", image, 0, 0, 6, 4]])
    raster.dispose()
    expect([raster.source.width, raster.source.height]).toEqual([0, 0])
  })
})

describe("PNG encoding", () => {
  it("encodes a layer as it is, and a mask as white with the mask's alpha", async () => {
    const raster = new CanvasRaster(6, 4)
    await expect(raster.toPng(false)).resolves.toBe(blobResult)
    expect(vi.mocked(HTMLCanvasElement.prototype.toBlob).mock.contexts[0]).toBe(raster.source)

    await expect(raster.toPng(true)).resolves.toBe(blobResult)
    const white = vi.mocked(HTMLCanvasElement.prototype.toBlob).mock.contexts[1] as HTMLCanvasElement
    expect(white).not.toBe(raster.source)
    expect(contextOf(white).calls).toEqual([["fillStyle=", "#ffffff"], ["fillRect", 0, 0, 6, 4], ["globalCompositeOperation=", "destination-in"], ["drawImage", raster.source, 0, 0]])
    expect([white.width, white.height]).toEqual([0, 0]) // freed after encoding
  })

  it("rejects when the browser cannot encode", async () => {
    blobResult = null
    await expect(new CanvasRaster(2, 2).toPng(true)).rejects.toThrow("PNG encoding failed")
  })

  it("encodes raw RGBA pixels and frees the canvas, also on failure", async () => {
    await expect(encodePng(new Uint8ClampedArray(2 * 3 * 4), 2, 3)).resolves.toBe(blobResult)
    const canvas = vi.mocked(HTMLCanvasElement.prototype.toBlob).mock.contexts[0] as HTMLCanvasElement
    expect(contextOf(canvas).calls).toEqual([["putImageData", 2, 3, 0, 0]])
    expect([canvas.width, canvas.height]).toEqual([0, 0])
    blobResult = null
    await expect(encodePng(new Uint8ClampedArray(4), 1, 1)).rejects.toThrow("PNG encoding failed")
    const failed = vi.mocked(HTMLCanvasElement.prototype.toBlob).mock.contexts[1] as HTMLCanvasElement
    expect([failed.width, failed.height]).toEqual([0, 0])
  })
})
