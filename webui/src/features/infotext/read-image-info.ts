/**
 * Reads generation parameters embedded in an image, in the browser: PNG text
 * chunks (`tEXt`, `iTXt`, `zTXt`; key "parameters"), and the EXIF
 * UserComment that A1111/Forge write into JPEG and WebP files. Returns null
 * when the image carries none.
 */

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
const PARAMETER_KEYS = ["parameters", "Parameters"]

const latin1 = new TextDecoder("latin1")
const utf8 = new TextDecoder("utf-8")

/** Inflated text larger than this is ignored (a deflate bomb must not take the tab down). */
export const MAX_INFLATED_BYTES = 4 * 1024 * 1024

async function inflate(bytes: Uint8Array, limit = MAX_INFLATED_BYTES): Promise<Uint8Array> {
  // PNG compression method 0 is zlib ("deflate" in the Compression Streams spec).
  const reader = new Response(bytes as BodyInit).body!.pipeThrough(new DecompressionStream("deflate")).getReader()
  const chunks: Uint8Array[] = []
  let size = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    size += value.byteLength
    if (size > limit) {
      await reader.cancel()
      throw new RangeError("inflated text too large")
    }
    chunks.push(value)
  }
  const out = new Uint8Array(size)
  let at = 0
  for (const chunk of chunks) {
    out.set(chunk, at)
    at += chunk.byteLength
  }
  return out
}

/**
 * Text chunks of a PNG by keyword (first occurrence wins). Compressed chunks
 * are only inflated for `keys` (all when omitted), each up to `MAX_INFLATED_BYTES`.
 */
export async function readPngText(bytes: Uint8Array, keys?: readonly string[]): Promise<Record<string, string>> {
  const out: Record<string, string> = {}
  if (!PNG_SIGNATURE.every((b, i) => bytes[i] === b)) return out
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  let at = 8
  while (at + 8 <= bytes.length) {
    const length = view.getUint32(at)
    const type = latin1.decode(bytes.subarray(at + 4, at + 8))
    const data = bytes.subarray(at + 8, Math.min(at + 8 + length, bytes.length))
    at += 12 + length
    if (type === "IEND") break
    if (type !== "tEXt" && type !== "iTXt" && type !== "zTXt") continue
    const nul = data.indexOf(0)
    if (nul <= 0) continue
    const key = latin1.decode(data.subarray(0, nul))
    if (key in out || (keys && !keys.includes(key))) continue
    try {
      if (type === "tEXt") {
        out[key] = latin1.decode(data.subarray(nul + 1))
      } else if (type === "zTXt") {
        out[key] = latin1.decode(await inflate(data.subarray(nul + 2)))
      } else {
        const compressed = data[nul + 1] === 1
        let p = nul + 3
        p = data.indexOf(0, p) + 1 // language tag
        p = data.indexOf(0, p) + 1 // translated keyword
        if (p <= 0) continue
        const text = data.subarray(p)
        out[key] = utf8.decode(compressed ? await inflate(text) : text)
      }
    } catch {
      // A broken chunk does not hide the others.
    }
  }
  return out
}

/** EXIF UserComment (tag 0x9286) from a TIFF block ("II*\0" / "MM\0*"). */
export function readExifUserComment(tiff: Uint8Array): string | null {
  if (tiff.length < 8) return null
  const view = new DataView(tiff.buffer, tiff.byteOffset, tiff.byteLength)
  const little = tiff[0] === 0x49
  const u16 = (at: number) => view.getUint16(at, little)
  const u32 = (at: number) => view.getUint32(at, little)

  const findTag = (ifd: number, tag: number): { type: number; count: number; valueAt: number } | null => {
    if (ifd + 2 > tiff.length) return null
    const entries = u16(ifd)
    for (let i = 0; i < entries; i++) {
      const entry = ifd + 2 + i * 12
      if (entry + 12 > tiff.length) return null
      if (u16(entry) !== tag) continue
      const type = u16(entry + 2)
      const count = u32(entry + 4)
      const size = type === 7 || type === 1 || type === 2 ? count : type === 3 ? count * 2 : count * 4
      return { type, count, valueAt: size <= 4 ? entry + 8 : u32(entry + 8) }
    }
    return null
  }

  const exifPointer = findTag(u32(4), 0x8769)
  if (!exifPointer) return null
  const comment = findTag(u32(exifPointer.valueAt), 0x9286)
  if (!comment || comment.valueAt + comment.count > tiff.length) return null
  const raw = tiff.subarray(comment.valueAt, comment.valueAt + comment.count)
  const prefix = latin1.decode(raw.subarray(0, 8))
  const body = raw.subarray(8)
  let text: string
  if (prefix.startsWith("UNICODE")) {
    // piexif writes UTF-16BE; other tools use LE. Guess from the first code unit.
    const bigEndian = body.length >= 2 && body[0] === 0 && body[1] !== 0
    text = new TextDecoder(bigEndian ? "utf-16be" : "utf-16le").decode(body)
  } else if (prefix.startsWith("ASCII")) {
    text = latin1.decode(body)
  } else {
    text = utf8.decode(prefix.startsWith("\0\0\0\0") ? body : raw)
  }
  text = text.replace(/\0+$/, "")
  return text || null
}

function jpegExif(bytes: Uint8Array): Uint8Array | null {
  if (bytes[0] !== 0xff || bytes[1] !== 0xd8) return null
  let at = 2
  while (at + 4 <= bytes.length && bytes[at] === 0xff) {
    const marker = bytes[at + 1]!
    const length = (bytes[at + 2]! << 8) | bytes[at + 3]!
    if (marker === 0xda || marker === 0xd9) break
    const segment = bytes.subarray(at + 4, at + 2 + length)
    if (marker === 0xe1 && latin1.decode(segment.subarray(0, 6)) === "Exif\0\0") return segment.subarray(6)
    at += 2 + length
  }
  return null
}

function webpExif(bytes: Uint8Array): Uint8Array | null {
  if (latin1.decode(bytes.subarray(0, 4)) !== "RIFF" || latin1.decode(bytes.subarray(8, 12)) !== "WEBP") return null
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  let at = 12
  while (at + 8 <= bytes.length) {
    const type = latin1.decode(bytes.subarray(at, at + 4))
    const size = view.getUint32(at + 4, true)
    if (type === "EXIF") {
      const data = bytes.subarray(at + 8, at + 8 + size)
      // Some writers keep the JPEG "Exif\0\0" header.
      return latin1.decode(data.subarray(0, 6)) === "Exif\0\0" ? data.subarray(6) : data
    }
    at += 8 + size + (size % 2)
  }
  return null
}

/** The generation parameters of an image file, or null. */
export async function readImageParameters(file: Blob): Promise<string | null> {
  const bytes = new Uint8Array(await file.arrayBuffer())
  const text = await readPngText(bytes, PARAMETER_KEYS)
  for (const key of PARAMETER_KEYS) {
    if (text[key]) return text[key]
  }
  const exif = jpegExif(bytes) ?? webpExif(bytes)
  return exif ? readExifUserComment(exif) : null
}
