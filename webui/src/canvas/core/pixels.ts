/**
 * Pixel transforms of the layer exports and imports (pure, on RGBA data as
 * `getImageData` returns it: unpremultiplied, row-major).
 *
 * Layer formats on the wire (schema-format.md §1.3):
 * - sketch: RGBA colour, transparent where nothing was painted;
 * - mask: alpha-coded, alpha 255 = masked. The editor exports white pixels
 *   with the mask's alpha, so the server reads the same mask whether it looks
 *   at the alpha or (a fully painted mask is opaque) at the luminance.
 */
import type { Rect } from "./geometry"
import { EMPTY_RECT } from "./geometry"

export type Rgb = readonly [number, number, number]

/**
 * Turns a mask layer's pixels (any display tint) into the export format in
 * place: RGB white, alpha kept. Returns whether anything is masked.
 */
export function toMaskExport(rgba: Uint8ClampedArray): boolean {
  let any = false
  for (let i = 0; i < rgba.length; i += 4) {
    rgba[i] = 255
    rgba[i + 1] = 255
    rgba[i + 2] = 255
    if (rgba[i + 3] !== 0) any = true
  }
  return any
}

/** Whether any pixel is not fully transparent. */
export function hasContent(rgba: ArrayLike<number>): boolean {
  for (let i = 3; i < rgba.length; i += 4) if (rgba[i] !== 0) return true
  return false
}

/**
 * The mask an image encodes, as alpha bytes (server `mask_channel`): its
 * alpha when some pixel is not opaque, else its luminance (white = masked,
 * a plain black-and-white mask image).
 */
export function maskAlphaFromImage(rgba: ArrayLike<number>): Uint8Array {
  const count = rgba.length >> 2
  const out = new Uint8Array(count)
  let opaque = true
  for (let i = 3; i < rgba.length; i += 4) {
    if (rgba[i] !== 255) {
      opaque = false
      break
    }
  }
  for (let i = 0, j = 0; i < count; i++, j += 4) {
    // ITU-R 601-2 luma with PIL's fixed-point rounding (convert("L")).
    out[i] = opaque ? (rgba[j]! * 19595 + rgba[j + 1]! * 38470 + rgba[j + 2]! * 7471 + 0x8000) >> 16 : rgba[j + 3]!
  }
  return out
}

/** RGBA pixels of alpha bytes in one colour. */
export function tintAlpha(alpha: ArrayLike<number>, [r, g, b]: Rgb): Uint8ClampedArray {
  const out = new Uint8ClampedArray(alpha.length * 4)
  for (let i = 0, j = 0; i < alpha.length; i++, j += 4) {
    out[j] = r
    out[j + 1] = g
    out[j + 2] = b
    out[j + 3] = alpha[i]!
  }
  return out
}

/** Bounding box of the pixels whose alpha is above `threshold` (empty rect when none). */
export function alphaBounds(rgba: ArrayLike<number>, width: number, height: number, threshold = 0): Rect {
  let x0 = width
  let y0 = height
  let x1 = -1
  let y1 = -1
  for (let y = 0; y < height; y++) {
    const row = y * width * 4
    let rowHas = false
    for (let x = 0; x < width; x++) {
      if (rgba[row + x * 4 + 3]! > threshold) {
        rowHas = true
        if (x < x0) x0 = x
        if (x > x1) x1 = x
      }
    }
    if (rowHas) {
      if (y < y0) y0 = y
      y1 = y
    }
  }
  return x1 < 0 ? EMPTY_RECT : { x: x0, y: y0, width: x1 - x0 + 1, height: y1 - y0 + 1 }
}

/**
 * Fast non-cryptographic 53-bit hash of bytes (cyrb53 over 32-bit words):
 * identifies a layer's content to reuse its earlier upload. `crypto.subtle`
 * is unavailable outside secure contexts (a LAN address over http).
 */
export function hashBytes(bytes: Uint8Array | Uint8ClampedArray, seed = 0): string {
  let h1 = 0xdeadbeef ^ seed
  let h2 = 0x41c6ce57 ^ seed
  const words = bytes.byteLength >> 2
  const view = bytes.byteOffset % 4 === 0 ? new Uint32Array(bytes.buffer, bytes.byteOffset, words) : null
  for (let i = 0; i < words; i++) {
    const w = view ? view[i]! : bytes[i * 4]! | (bytes[i * 4 + 1]! << 8) | (bytes[i * 4 + 2]! << 16) | (bytes[i * 4 + 3]! << 24)
    h1 = Math.imul(h1 ^ w, 2654435761)
    h2 = Math.imul(h2 ^ w, 1597334677)
  }
  for (let i = words * 4; i < bytes.byteLength; i++) {
    h1 = Math.imul(h1 ^ bytes[i]!, 2654435761)
    h2 = Math.imul(h2 ^ bytes[i]!, 1597334677)
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909)
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909)
  return `${(h2 >>> 0).toString(16).padStart(8, "0")}${(h1 >>> 0).toString(16).padStart(8, "0")}:${bytes.byteLength}`
}

/** `#rgb` / `#rrggbb` (/`#rrggbbaa`, alpha ignored) → RGB; null when malformed. */
export function parseHex(color: string): Rgb | null {
  const hex = color.trim().replace(/^#/, "")
  if (/^[0-9a-f]{3}$/i.test(hex)) {
    const [r, g, b] = [...hex].map((c) => parseInt(c + c, 16))
    return [r!, g!, b!]
  }
  if (/^[0-9a-f]{6}([0-9a-f]{2})?$/i.test(hex)) return [parseInt(hex.slice(0, 2), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(4, 6), 16)]
  return null
}

export function toHex([r, g, b]: Rgb): string {
  return `#${[r, g, b].map((v) => Math.round(v).toString(16).padStart(2, "0")).join("")}`
}

/** `src` over `dst` (both unpremultiplied RGBA, alpha 0–255) → RGBA. */
export function blendOver(dst: readonly number[], src: readonly number[]): [number, number, number, number] {
  const sa = src[3]! / 255
  const da = dst[3]! / 255
  const a = sa + da * (1 - sa)
  if (a <= 0) return [0, 0, 0, 0]
  const channel = (i: number) => Math.round((src[i]! * sa + dst[i]! * da * (1 - sa)) / a)
  return [channel(0), channel(1), channel(2), Math.round(a * 255)]
}
