/**
 * Compact pixel patches for the undo history (pure). A patch is the content
 * of a rectangle of a layer before an edit:
 *
 * - `empty`: fully transparent (painting on an empty area, the common case);
 * - `rle`: run-length encoded 32-bit pixels (`rgba`) or alpha bytes (`alpha`,
 *   for masks, whose colour is the display tint) — flat sketches and masks
 *   compress to a few runs per row;
 * - `raw`: the pixels as they are when runs would not save enough.
 */

export type PatchFormat = "rgba" | "alpha"

export type Patch =
  | { kind: "empty"; width: number; height: number }
  | { kind: "raw"; format: PatchFormat; width: number; height: number; data: Uint8Array | Uint8ClampedArray }
  | { kind: "rle"; format: PatchFormat; width: number; height: number; runs: Uint32Array }

/** Fixed cost of a patch object, so thousands of tiny patches still count. */
const OVERHEAD = 64

export function patchBytes(patch: Patch): number {
  switch (patch.kind) {
    case "empty":
      return OVERHEAD
    case "raw":
      return OVERHEAD + patch.data.byteLength
    case "rle":
      return OVERHEAD + patch.runs.byteLength
  }
}

/** Whether every pixel of RGBA data has alpha 0. */
export function isTransparent(rgba: ArrayLike<number>): boolean {
  for (let i = 3; i < rgba.length; i += 4) if (rgba[i] !== 0) return false
  return true
}

function pixelWords(rgba: Uint8ClampedArray | Uint8Array): Uint32Array {
  if (rgba.byteOffset % 4 === 0) return new Uint32Array(rgba.buffer, rgba.byteOffset, rgba.byteLength >> 2)
  return new Uint32Array(rgba.slice().buffer)
}

function alphaBytes(rgba: ArrayLike<number>): Uint8Array {
  const out = new Uint8Array(rgba.length >> 2)
  for (let i = 0, j = 3; i < out.length; i++, j += 4) out[i] = rgba[j]!
  return out
}

/** Number of runs of equal values. */
function countRuns(values: ArrayLike<number>): number {
  if (values.length === 0) return 0
  let runs = 1
  for (let i = 1; i < values.length; i++) if (values[i] !== values[i - 1]) runs++
  return runs
}

function encodeRuns(values: ArrayLike<number>, runs: number): Uint32Array {
  const out = new Uint32Array(runs * 2)
  let k = 0
  let current = values[0]!
  let count = 1
  for (let i = 1; i < values.length; i++) {
    const value = values[i]!
    if (value === current) {
      count++
      continue
    }
    out[k++] = current
    out[k++] = count
    current = value
    count = 1
  }
  out[k++] = current
  out[k] = count
  return out
}

/**
 * Encodes the RGBA pixels of a `width × height` rectangle. `rgba` may be
 * kept by the patch (pass a fresh array, e.g. `getImageData().data`).
 */
export function encodePatch(rgba: Uint8ClampedArray, width: number, height: number, format: PatchFormat): Patch {
  if (rgba.length !== width * height * 4) throw new Error(`patch data is ${rgba.length} bytes for ${width}×${height}`)
  if (isTransparent(rgba)) return { kind: "empty", width, height }
  const values = format === "alpha" ? alphaBytes(rgba) : pixelWords(rgba)
  const runs = countRuns(values)
  const rawBytes = format === "alpha" ? values.length : rgba.byteLength
  // A run costs 8 bytes; keep the raw pixels unless runs save at least a quarter.
  if (runs * 8 < rawBytes * 0.75) return { kind: "rle", format, width, height, runs: encodeRuns(values, runs) }
  return { kind: "raw", format, width, height, data: format === "alpha" ? (values as Uint8Array) : rgba }
}

export type Tint = readonly [number, number, number]

/** Writes alpha values as RGBA with the tint colour. */
function writeAlpha(out: Uint8ClampedArray, index: number, alpha: number, tint: Tint) {
  const j = index * 4
  out[j] = tint[0]
  out[j + 1] = tint[1]
  out[j + 2] = tint[2]
  out[j + 3] = alpha
}

/** RGBA pixels of a patch; alpha patches take the `tint` colour (default white). */
export function decodePatch(patch: Patch, tint: Tint = [255, 255, 255]): Uint8ClampedArray {
  const count = patch.width * patch.height
  if (patch.kind === "empty") return new Uint8ClampedArray(count * 4)
  if (patch.kind === "raw") {
    if (patch.format === "rgba") return patch.data instanceof Uint8ClampedArray ? patch.data : new Uint8ClampedArray(patch.data)
    const out = new Uint8ClampedArray(count * 4)
    for (let i = 0; i < count; i++) writeAlpha(out, i, patch.data[i]!, tint)
    return out
  }
  const out = new Uint8ClampedArray(count * 4)
  const words = patch.format === "rgba" ? new Uint32Array(out.buffer) : null
  let index = 0
  for (let k = 0; k + 1 < patch.runs.length; k += 2) {
    const value = patch.runs[k]!
    const end = Math.min(count, index + patch.runs[k + 1]!)
    if (words) words.fill(value, index, end)
    else for (let i = index; i < end; i++) writeAlpha(out, i, value, tint)
    index = end
  }
  return out
}
