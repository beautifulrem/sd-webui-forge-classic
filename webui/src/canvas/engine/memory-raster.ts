/**
 * A raster on a typed array, with the same compositing rules as the canvas
 * (source-over, destination-out with global alpha). Dabs and polygons are
 * rasterised without antialiasing. Used by tests (jsdom has no 2D canvas).
 */
import type { Rect } from "../core/geometry"
import type { Rgb } from "../core/pixels"
import type { Raster, RasterFactory } from "./raster"
import { HARD } from "./raster"

export class MemoryRaster implements Raster {
  readonly width: number
  readonly height: number
  readonly data: Uint8ClampedArray
  readonly source = null

  constructor(width: number, height: number) {
    this.width = width
    this.height = height
    this.data = new Uint8ClampedArray(width * height * 4)
  }

  read(rect: Rect): Uint8ClampedArray {
    const out = new Uint8ClampedArray(rect.width * rect.height * 4)
    for (let y = 0; y < rect.height; y++) {
      const from = ((rect.y + y) * this.width + rect.x) * 4
      out.set(this.data.subarray(from, from + rect.width * 4), y * rect.width * 4)
    }
    return out
  }

  write(rect: Rect, data: Uint8ClampedArray) {
    for (let y = 0; y < rect.height; y++) {
      this.data.set(data.subarray(y * rect.width * 4, (y + 1) * rect.width * 4), ((rect.y + y) * this.width + rect.x) * 4)
    }
  }

  clear(rect?: Rect) {
    if (!rect) {
      this.data.fill(0)
      return
    }
    for (let y = rect.y; y < rect.y + rect.height; y++) this.data.fill(0, (y * this.width + rect.x) * 4, (y * this.width + rect.x + rect.width) * 4)
  }

  /** Source-over of one colour at coverage `a` (0–1). */
  private paint(index: number, color: Rgb, a: number) {
    const j = index * 4
    const da = this.data[j + 3]! / 255
    const out = a + da * (1 - a)
    if (out <= 0) return
    for (let c = 0; c < 3; c++) this.data[j + c] = Math.round((color[c]! * a + this.data[j + c]! * da * (1 - a)) / out)
    this.data[j + 3] = Math.round(out * 255)
  }

  dabs(dabs: ArrayLike<number>, hardness: number, color: Rgb) {
    for (let i = 0; i + 2 < dabs.length; i += 3) {
      const cx = dabs[i]!
      const cy = dabs[i + 1]!
      const r = dabs[i + 2]!
      for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(this.height - 1, Math.ceil(cy + r)); y++) {
        for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(this.width - 1, Math.ceil(cx + r)); x++) {
          const d = Math.hypot(x + 0.5 - cx, y + 0.5 - cy) / r
          if (d > 1) continue
          const inner = Math.min(hardness, 0.999)
          const a = hardness >= HARD || d <= inner ? 1 : 0.5 + 0.5 * Math.cos((Math.PI * (d - inner)) / (1 - inner))
          if (a > 0) this.paint(y * this.width + x, color, a)
        }
      }
    }
  }

  polygon(points: ArrayLike<number>, color: Rgb) {
    const n = points.length >> 1
    if (n < 3) return
    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        // Even-odd crossing test at the pixel centre (enough for the simple shapes tests draw).
        const px = x + 0.5
        const py = y + 0.5
        let inside = false
        for (let i = 0, j = n - 1; i < n; j = i++) {
          const xi = points[i * 2]!
          const yi = points[i * 2 + 1]!
          const xj = points[j * 2]!
          const yj = points[j * 2 + 1]!
          if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) inside = !inside
        }
        if (inside) this.paint(y * this.width + x, color, 1)
      }
    }
  }

  composite(src: Raster, rect: Rect, opacity: number, erase: boolean) {
    const pixels = src.read(rect)
    for (let y = 0; y < rect.height; y++) {
      for (let x = 0; x < rect.width; x++) {
        const s = (y * rect.width + x) * 4
        const sa = (pixels[s + 3]! / 255) * opacity
        if (sa <= 0) continue
        const index = (rect.y + y) * this.width + rect.x + x
        if (erase) this.data[index * 4 + 3] = Math.round(this.data[index * 4 + 3]! * (1 - sa))
        else this.paint(index, [pixels[s]!, pixels[s + 1]!, pixels[s + 2]!], sa)
      }
    }
  }

  tint([r, g, b]: Rgb) {
    for (let j = 0; j < this.data.length; j += 4) {
      this.data[j] = r
      this.data[j + 1] = g
      this.data[j + 2] = b
    }
  }

  invert(color: Rgb) {
    this.tint(color)
    for (let j = 3; j < this.data.length; j += 4) this.data[j] = 255 - this.data[j]!
  }

  /** Draws a `TestImage` of the raster's size (tests have no image decoding). */
  drawImage(source: CanvasImageSource) {
    const pixels = (source as Partial<TestImage>).pixels
    if (!pixels || pixels.length !== this.data.length) throw new Error("MemoryRaster draws only TestImages of its size")
    this.data.set(pixels)
  }
}

/** A stand-in for a decoded image in tests: its size and RGBA pixels. */
export type TestImage = { naturalWidth: number; naturalHeight: number; pixels: Uint8ClampedArray }

/** A `TestImage` filled with one RGBA colour. */
export function testImage(width: number, height: number, rgba: readonly [number, number, number, number] = [255, 255, 255, 255]): TestImage & CanvasImageSource {
  const pixels = new Uint8ClampedArray(width * height * 4)
  for (let i = 0; i < pixels.length; i += 4) pixels.set(rgba, i)
  return { naturalWidth: width, naturalHeight: height, pixels } as unknown as TestImage & CanvasImageSource
}

export const createMemoryRaster: RasterFactory = (width, height) => new MemoryRaster(width, height)
