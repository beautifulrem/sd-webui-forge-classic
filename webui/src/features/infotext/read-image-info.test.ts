import { describe, expect, it } from "vitest"
import { MAX_INFLATED_BYTES, readExifUserComment, readImageParameters, readPngText } from "./read-image-info"

async function deflate(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Response(bytes as BodyInit).body!.pipeThrough(new CompressionStream("deflate"))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

const latin1 = (text: string) => Uint8Array.from(text, (c) => c.charCodeAt(0))
const utf8 = (text: string) => new TextEncoder().encode(text)
const concat = (...parts: Uint8Array[]) => {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0))
  let at = 0
  for (const part of parts) {
    out.set(part, at)
    at += part.length
  }
  return out
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const head = new Uint8Array(8)
  new DataView(head.buffer).setUint32(0, data.length)
  head.set(latin1(type), 4)
  return concat(head, data, new Uint8Array(4)) // CRC is not checked
}

const SIGNATURE = Uint8Array.of(0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a)
const png = (...chunks: Uint8Array[]) => concat(SIGNATURE, chunk("IHDR", new Uint8Array(13)), ...chunks, chunk("IEND", new Uint8Array()))

const PARAMS = "1girl, 笑顔\nNegative prompt: lowres\nSteps: 20, Seed: 42"

describe("readPngText", () => {
  it("reads tEXt, iTXt (plain and compressed) and zTXt chunks", async () => {
    const bytes = png(
      chunk("tEXt", concat(latin1("Software\0"), latin1("Remi"))),
      chunk("iTXt", concat(latin1("parameters\0"), Uint8Array.of(0, 0), latin1("\0\0"), utf8(PARAMS))),
      chunk("iTXt", concat(latin1("comment\0"), Uint8Array.of(1, 0), latin1("en\0Comment\0"), await deflate(utf8("zipped ✓")))),
      chunk("zTXt", concat(latin1("legacy\0"), Uint8Array.of(0), await deflate(latin1("old text")))),
    )
    expect(await readPngText(bytes)).toEqual({ Software: "Remi", parameters: PARAMS, comment: "zipped ✓", legacy: "old text" })
  })

  it("inflates only the wanted keys, and refuses oversized text", async () => {
    const bomb = await deflate(new Uint8Array(MAX_INFLATED_BYTES + 1))
    const bytes = png(
      chunk("zTXt", concat(latin1("other\0"), Uint8Array.of(0), bomb)),
      chunk("zTXt", concat(latin1("parameters\0"), Uint8Array.of(0), bomb)),
      chunk("tEXt", concat(latin1("parameters\0"), latin1("fallback"))),
    )
    // The oversized chunk is skipped; the next one with the key is read.
    expect(await readPngText(bytes, ["parameters"])).toEqual({ parameters: "fallback" })
  })

  it("returns nothing for a non-PNG", async () => {
    expect(await readPngText(latin1("GIF89a"))).toEqual({})
  })
})

/** A little-endian TIFF block with IFD0 → Exif IFD → UserComment. */
function tiffWithComment(comment: Uint8Array): Uint8Array {
  const header = 8
  const ifd0 = header // 2 + 12 + 4 = 18 bytes
  const exifIfd = ifd0 + 18 // 18 bytes
  const value = exifIfd + 18
  const out = new Uint8Array(value + comment.length)
  const v = new DataView(out.buffer)
  out.set(latin1("II*\0"), 0)
  v.setUint32(4, ifd0, true)
  v.setUint16(ifd0, 1, true)
  v.setUint16(ifd0 + 2, 0x8769, true)
  v.setUint16(ifd0 + 4, 4, true)
  v.setUint32(ifd0 + 6, 1, true)
  v.setUint32(ifd0 + 10, exifIfd, true)
  v.setUint16(exifIfd, 1, true)
  v.setUint16(exifIfd + 2, 0x9286, true)
  v.setUint16(exifIfd + 4, 7, true)
  v.setUint32(exifIfd + 6, comment.length, true)
  v.setUint32(exifIfd + 10, value, true)
  out.set(comment, value)
  return out
}

function utf16be(text: string): Uint8Array {
  const out = new Uint8Array(text.length * 2)
  for (let i = 0; i < text.length; i++) {
    out[i * 2] = text.charCodeAt(i) >> 8
    out[i * 2 + 1] = text.charCodeAt(i) & 0xff
  }
  return out
}

describe("EXIF UserComment", () => {
  it("decodes piexif's UNICODE (UTF-16BE) comments", () => {
    expect(readExifUserComment(tiffWithComment(concat(latin1("UNICODE\0"), utf16be(PARAMS))))).toBe(PARAMS)
  })

  it("decodes ASCII comments", () => {
    expect(readExifUserComment(tiffWithComment(concat(latin1("ASCII\0\0\0"), latin1("Steps: 5"))))).toBe("Steps: 5")
  })

  it("finds the comment in a JPEG APP1 segment", async () => {
    const exif = concat(latin1("Exif\0\0"), tiffWithComment(concat(latin1("UNICODE\0"), utf16be(PARAMS))))
    const app1 = concat(Uint8Array.of(0xff, 0xe1, (exif.length + 2) >> 8, (exif.length + 2) & 0xff), exif)
    const jpeg = concat(Uint8Array.of(0xff, 0xd8), app1, Uint8Array.of(0xff, 0xda, 0, 2, 0xff, 0xd9))
    expect(await readImageParameters(new Blob([jpeg]))).toBe(PARAMS)
  })

  it("finds the comment in a WebP EXIF chunk", async () => {
    const exif = tiffWithComment(concat(latin1("UNICODE\0"), utf16be("Steps: 8")))
    const size = new Uint8Array(4)
    new DataView(size.buffer).setUint32(0, exif.length, true)
    const body = concat(latin1("WEBP"), latin1("EXIF"), size, exif, exif.length % 2 ? new Uint8Array(1) : new Uint8Array())
    const riffSize = new Uint8Array(4)
    new DataView(riffSize.buffer).setUint32(0, body.length, true)
    expect(await readImageParameters(new Blob([concat(latin1("RIFF"), riffSize, body)]))).toBe("Steps: 8")
  })

  it("prefers the PNG parameters chunk", async () => {
    expect(await readImageParameters(new Blob([png(chunk("tEXt", concat(latin1("parameters\0"), latin1("Steps: 3"))))]))).toBe("Steps: 3")
    expect(await readImageParameters(new Blob([png()]))).toBeNull()
  })
})
