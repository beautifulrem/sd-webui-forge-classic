import { describe, expect, it } from "vitest"
import { normalizeEvent } from "./events"
import { encodeFrame, FrameError, FrameFormat, FRAME_HEADER_SIZE, parseFrame } from "./frames"
import { backoffDelay } from "./ws"

describe("parseFrame", () => {
  it("reads the little-endian header and the image bytes", () => {
    const bytes = new Uint8Array([0xff, 0xd8, 0xff, 0xe0])
    const frame = parseFrame(encodeFrame(FrameFormat.Jpeg, 0x01020304, 17, bytes))
    expect(frame).toMatchObject({ type: 1, mime: "image/jpeg", jobSeq: 0x01020304, step: 17 })
    expect([...frame.bytes]).toEqual([...bytes])
  })

  it("matches a hand-built buffer byte for byte", () => {
    // type=1, fmt=2 (webp), job_seq=258 (02 01 00 00), step=3 (03 00 00 00), payload "RIFF"
    const raw = new Uint8Array([1, 2, 2, 1, 0, 0, 3, 0, 0, 0, 0x52, 0x49, 0x46, 0x46])
    const frame = parseFrame(raw.buffer)
    expect(frame.mime).toBe("image/webp")
    expect(frame.jobSeq).toBe(258)
    expect(frame.step).toBe(3)
    expect(new TextDecoder().decode(frame.bytes)).toBe("RIFF")
  })

  it("handles the largest u32 values", () => {
    const frame = parseFrame(encodeFrame(FrameFormat.Webp, 0xffffffff, 0xfffffffe, new Uint8Array()))
    expect(frame.jobSeq).toBe(4294967295)
    expect(frame.step).toBe(4294967294)
    expect(frame.bytes.byteLength).toBe(0)
  })

  it("rejects short, unknown-type and unknown-format frames", () => {
    expect(() => parseFrame(new ArrayBuffer(FRAME_HEADER_SIZE - 1))).toThrow(FrameError)
    const badType = new Uint8Array(FRAME_HEADER_SIZE)
    badType[0] = 9
    badType[1] = 1
    expect(() => parseFrame(badType.buffer)).toThrow(/type 9/)
    const badFormat = new Uint8Array(FRAME_HEADER_SIZE)
    badFormat[0] = 1
    badFormat[1] = 7
    expect(() => parseFrame(badFormat.buffer)).toThrow(/format 7/)
  })
})

describe("normalizeEvent", () => {
  it("accepts flat and enveloped events", () => {
    expect(normalizeEvent({ type: "job.progress", job_id: "a", step: 1, steps: 2 })).toEqual({
      type: "job.progress",
      job_id: "a",
      step: 1,
      steps: 2,
    })
    expect(normalizeEvent({ type: "model.loading", data: { stage: "loading", model: "m" } })).toEqual({
      type: "model.loading",
      stage: "loading",
      model: "m",
    })
  })

  it("rejects non-events", () => {
    expect(normalizeEvent(null)).toBeNull()
    expect(normalizeEvent({ kind: "x" })).toBeNull()
    expect(normalizeEvent("hello")).toBeNull()
  })
})

describe("backoffDelay", () => {
  it("grows exponentially with jitter and caps at 15 s", () => {
    expect(backoffDelay(0, () => 0.5)).toBe(500)
    expect(backoffDelay(3, () => 0.5)).toBe(4000)
    expect(backoffDelay(20, () => 0.5)).toBe(15000)
    expect(backoffDelay(0, () => 0)).toBe(375)
    expect(backoffDelay(0, () => 1)).toBe(625)
  })
})
