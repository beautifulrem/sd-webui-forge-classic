/**
 * Binary WebSocket frames. Layout (little-endian):
 *
 *   u8 type   1 = live preview
 *   u8 fmt    1 = jpeg, 2 = webp
 *   u32 job_seq
 *   u32 step
 *   ...image bytes
 */

export const FRAME_HEADER_SIZE = 10

export const FrameType = { Preview: 1 } as const
export const FrameFormat = { Jpeg: 1, Webp: 2 } as const

const MIME: Record<number, string> = {
  [FrameFormat.Jpeg]: "image/jpeg",
  [FrameFormat.Webp]: "image/webp",
}

export type PreviewFrame = {
  type: typeof FrameType.Preview
  mime: string
  jobSeq: number
  step: number
  bytes: Uint8Array<ArrayBuffer>
}

export class FrameError extends Error {}

/** Parses one binary frame; throws `FrameError` on malformed or unknown frames. */
export function parseFrame(buffer: ArrayBuffer): PreviewFrame {
  if (buffer.byteLength < FRAME_HEADER_SIZE) throw new FrameError(`frame too short (${buffer.byteLength} bytes)`)
  const view = new DataView(buffer)
  const type = view.getUint8(0)
  if (type !== FrameType.Preview) throw new FrameError(`unknown frame type ${type}`)
  const fmt = view.getUint8(1)
  const mime = MIME[fmt]
  if (!mime) throw new FrameError(`unknown image format ${fmt}`)
  return {
    type,
    mime,
    jobSeq: view.getUint32(2, true),
    step: view.getUint32(6, true),
    bytes: new Uint8Array(buffer, FRAME_HEADER_SIZE),
  }
}

/** Builds a frame; used by the mock server and tests. */
export function encodeFrame(fmt: number, jobSeq: number, step: number, bytes: Uint8Array): ArrayBuffer {
  const buffer = new ArrayBuffer(FRAME_HEADER_SIZE + bytes.byteLength)
  const view = new DataView(buffer)
  view.setUint8(0, FrameType.Preview)
  view.setUint8(1, fmt)
  view.setUint32(2, jobSeq, true)
  view.setUint32(6, step, true)
  new Uint8Array(buffer, FRAME_HEADER_SIZE).set(bytes)
  return buffer
}
