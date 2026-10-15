/**
 * Pixel surfaces of the editor. A layer is one full-resolution raster (no
 * per-stroke objects); strokes and shapes are drawn into a scratch raster at
 * full strength and composited onto the layer once, at the brush opacity, so
 * overlapping dabs of one stroke never build up.
 *
 * `CanvasRaster` is the browser implementation (a 2D canvas, GPU-backed);
 * `MemoryRaster` (memory-raster.ts) implements the same operations on a
 * typed array for tests.
 */
import type { Rect } from "../core/geometry"
import type { Rgb } from "../core/pixels"
import { toHex } from "../core/pixels"

export interface Raster {
  readonly width: number
  readonly height: number
  /** RGBA pixels of `rect` (inside the raster), unpremultiplied. */
  read(rect: Rect): Uint8ClampedArray
  write(rect: Rect, data: Uint8ClampedArray): void
  clear(rect?: Rect): void
  /** Round dabs `[x, y, radius, …]` in an opaque colour; `hardness` 1 = hard edge, 0 = all falloff. */
  dabs(dabs: ArrayLike<number>, hardness: number, color: Rgb): void
  /** Fills a polygon `[x0, y0, x1, y1, …]` in an opaque colour. */
  polygon(points: ArrayLike<number>, color: Rgb): void
  /** `src`'s `rect` drawn onto this raster at `opacity`; `erase` removes instead (destination-out). */
  composite(src: Raster, rect: Rect, opacity: number, erase: boolean): void
  /** Recolours every pixel, keeping alpha (a mask's display tint). */
  tint(color: Rgb): void
  /** alpha → 255 − alpha, in `color` (inverts a mask). */
  invert(color: Rgb): void
  /** Draws an image scaled to the whole raster (replacing nothing; clear first). */
  drawImage(source: CanvasImageSource): void
  /** The raster as a PNG (`mask`: white with the raster's alpha), encoded off the pixel loop; absent on memory rasters. */
  toPng?(mask: boolean): Promise<Blob>
  /** What a renderer draws (null for memory rasters). */
  readonly source: HTMLCanvasElement | null
}

export type RasterFactory = (width: number, height: number, options?: { readFrequently?: boolean }) => Raster

/** Stamp textures of soft brushes, per hardness and colour. */
const stamps = new Map<string, HTMLCanvasElement>()
const STAMP_SIZE = 256

function softStamp(hardness: number, color: Rgb): HTMLCanvasElement {
  const key = `${hardness.toFixed(2)}|${toHex(color)}`
  const cached = stamps.get(key)
  if (cached) return cached
  const canvas = document.createElement("canvas")
  canvas.width = canvas.height = STAMP_SIZE
  const ctx = canvas.getContext("2d")!
  const r = STAMP_SIZE / 2
  const gradient = ctx.createRadialGradient(r, r, 0, r, r, r)
  const [red, green, blue] = color
  const rgba = (a: number) => `rgba(${red}, ${green}, ${blue}, ${a})`
  const inner = Math.max(0, Math.min(0.999, hardness))
  gradient.addColorStop(0, rgba(1))
  gradient.addColorStop(inner, rgba(1))
  // A smooth (cosine-like) falloff reads softer than a linear ramp.
  for (let i = 1; i <= 6; i++) {
    const t = i / 6
    gradient.addColorStop(inner + (1 - inner) * t, rgba(0.5 + 0.5 * Math.cos(Math.PI * t)))
  }
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, STAMP_SIZE, STAMP_SIZE)
  if (stamps.size >= 12) stamps.delete(stamps.keys().next().value!)
  stamps.set(key, canvas)
  return canvas
}

/** Hardness from which a dab is drawn as a plain antialiased disc. */
export const HARD = 0.98

export class CanvasRaster implements Raster {
  readonly width: number
  readonly height: number
  readonly source: HTMLCanvasElement
  private readonly ctx: CanvasRenderingContext2D

  constructor(width: number, height: number, { readFrequently = false }: { readFrequently?: boolean } = {}) {
    this.width = width
    this.height = height
    this.source = document.createElement("canvas")
    this.source.width = width
    this.source.height = height
    const ctx = this.source.getContext("2d", { willReadFrequently: readFrequently })
    if (!ctx) throw new Error("2D canvas unavailable")
    this.ctx = ctx
  }

  read(rect: Rect): Uint8ClampedArray {
    return this.ctx.getImageData(rect.x, rect.y, rect.width, rect.height).data
  }

  write(rect: Rect, data: Uint8ClampedArray) {
    this.ctx.putImageData(new ImageData(data as Uint8ClampedArray<ArrayBuffer>, rect.width, rect.height), rect.x, rect.y)
  }

  clear(rect?: Rect) {
    if (rect) this.ctx.clearRect(rect.x, rect.y, rect.width, rect.height)
    else this.ctx.clearRect(0, 0, this.width, this.height)
  }

  dabs(dabs: ArrayLike<number>, hardness: number, color: Rgb) {
    const ctx = this.ctx
    if (dabs.length < 3) return
    if (hardness >= HARD) {
      // One path of discs: their union is filled once, so edges are not darkened by overlaps.
      ctx.fillStyle = toHex(color)
      ctx.beginPath()
      for (let i = 0; i + 2 < dabs.length; i += 3) {
        const x = dabs[i]!
        const y = dabs[i + 1]!
        const r = dabs[i + 2]!
        ctx.moveTo(x + r, y)
        ctx.arc(x, y, r, 0, Math.PI * 2)
      }
      ctx.fill()
      return
    }
    const stamp = softStamp(hardness, color)
    for (let i = 0; i + 2 < dabs.length; i += 3) {
      const r = dabs[i + 2]!
      ctx.drawImage(stamp, dabs[i]! - r, dabs[i + 1]! - r, r * 2, r * 2)
    }
  }

  polygon(points: ArrayLike<number>, color: Rgb) {
    if (points.length < 6) return
    const ctx = this.ctx
    ctx.fillStyle = toHex(color)
    ctx.beginPath()
    ctx.moveTo(points[0]!, points[1]!)
    for (let i = 2; i + 1 < points.length; i += 2) ctx.lineTo(points[i]!, points[i + 1]!)
    ctx.closePath()
    ctx.fill("nonzero")
  }

  composite(src: Raster, rect: Rect, opacity: number, erase: boolean) {
    if (!src.source) throw new Error("composite needs a canvas raster")
    const ctx = this.ctx
    ctx.save()
    ctx.globalAlpha = Math.max(0, Math.min(1, opacity))
    ctx.globalCompositeOperation = erase ? "destination-out" : "source-over"
    ctx.drawImage(src.source, rect.x, rect.y, rect.width, rect.height, rect.x, rect.y, rect.width, rect.height)
    ctx.restore()
  }

  tint(color: Rgb) {
    const ctx = this.ctx
    ctx.save()
    ctx.globalCompositeOperation = "source-in"
    ctx.fillStyle = toHex(color)
    ctx.fillRect(0, 0, this.width, this.height)
    ctx.restore()
  }

  invert(color: Rgb) {
    const scratch = document.createElement("canvas")
    scratch.width = this.width
    scratch.height = this.height
    const s = scratch.getContext("2d")!
    s.fillStyle = toHex(color)
    s.fillRect(0, 0, this.width, this.height)
    s.globalCompositeOperation = "destination-out"
    s.drawImage(this.source, 0, 0)
    this.clear()
    this.ctx.drawImage(scratch, 0, 0)
    scratch.width = scratch.height = 0
  }

  drawImage(source: CanvasImageSource) {
    this.ctx.drawImage(source, 0, 0, this.width, this.height)
  }

  toPng(mask: boolean): Promise<Blob> {
    if (!mask) return canvasPng(this.source)
    // White, keeping the mask's alpha: (255, 255, 255, a) exactly, whatever the display tint.
    const canvas = document.createElement("canvas")
    canvas.width = this.width
    canvas.height = this.height
    const ctx = canvas.getContext("2d")!
    ctx.fillStyle = "#ffffff"
    ctx.fillRect(0, 0, this.width, this.height)
    ctx.globalCompositeOperation = "destination-in"
    ctx.drawImage(this.source, 0, 0)
    return canvasPng(canvas).finally(() => {
      canvas.width = canvas.height = 0
    })
  }

  /** Frees the backing store now instead of at garbage collection (large canvases). */
  dispose() {
    this.source.width = this.source.height = 0
  }
}

export const createCanvasRaster: RasterFactory = (width, height, options) => new CanvasRaster(width, height, options)

/** A canvas as PNG (its pixels are taken when called; the encoding runs off the main thread). */
function canvasPng(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise<Blob>((resolve, reject) => canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("PNG encoding failed"))), "image/png"))
}

/** PNG of RGBA pixels. */
export async function encodePng(data: Uint8ClampedArray, width: number, height: number): Promise<Blob> {
  const canvas = document.createElement("canvas")
  canvas.width = width
  canvas.height = height
  canvas.getContext("2d")!.putImageData(new ImageData(data as Uint8ClampedArray<ArrayBuffer>, width, height), 0, 0)
  try {
    return await canvasPng(canvas)
  } finally {
    canvas.width = canvas.height = 0
  }
}
