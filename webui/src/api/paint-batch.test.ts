import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

describe("paintBatch", () => {
  let frames: Map<number, FrameRequestCallback>
  let now = 0
  /** Runs the pending animation frame callbacks at `ms` later. */
  const frame = (ms = 16) => {
    now += ms
    const callbacks = [...frames.values()]
    frames.clear()
    for (const callback of callbacks) callback(now)
  }

  beforeEach(() => {
    vi.resetModules()
    frames = new Map()
    let id = 0
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
      frames.set(++id, callback)
      return id
    })
    vi.stubGlobal("cancelAnimationFrame", (handle: number) => void frames.delete(handle))
    vi.spyOn(document, "visibilityState", "get").mockReturnValue("visible")
  })
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it("applies the updates of one step together in the next animation frame", async () => {
    const { paintBatch } = await import("./paint-batch")
    const applied: string[] = []
    paintBatch(() => applied.push("progress"))
    paintBatch(() => applied.push("preview"))
    expect(applied).toEqual([])
    expect(frames.size).toBe(1)
    frame()
    expect(applied).toEqual(["progress", "preview"])
    expect(frames.size).toBe(0)
  })

  it("waits for a decoding preview so it paints with the step's progress", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] })
    try {
      const { holdPaint, paintBatch, MAX_HOLD_MS } = await import("./paint-batch")
      const applied: string[] = []
      paintBatch(() => applied.push("progress"))
      const release = holdPaint() // the preview arrived after its progress, before the frame ran
      frame()
      expect(applied).toEqual([])
      expect(frames.size).toBe(0) // no animation frames while holding (each would be drawn)
      paintBatch(() => applied.push("preview"))
      release()
      release() // twice: no effect
      expect(frames.size).toBe(1)
      frame()
      expect(applied).toEqual(["progress", "preview"])
      // a hold before anything is queued
      const next = holdPaint()
      paintBatch(() => applied.push("progress 2"))
      expect(frames.size).toBe(0)
      next()
      frame()
      expect(applied).toHaveLength(3)
      // a decode that never ends holds the batch for MAX_HOLD_MS at most
      holdPaint()
      paintBatch(() => applied.push("late"))
      vi.advanceTimersByTime(MAX_HOLD_MS - 1)
      expect(frames.size).toBe(0)
      vi.advanceTimersByTime(1)
      frame()
      expect(applied).toEqual(["progress", "preview", "progress 2", "late"])
    } finally {
      vi.useRealTimers()
    }
  })

  it("flushes on demand, and at once while the page is hidden", async () => {
    const { flushPaintBatch, paintBatch } = await import("./paint-batch")
    const applied: number[] = []
    paintBatch(() => applied.push(1))
    flushPaintBatch()
    expect(applied).toEqual([1])
    expect(frames.size).toBe(0)
    vi.spyOn(document, "visibilityState", "get").mockReturnValue("hidden")
    paintBatch(() => applied.push(2))
    expect(applied).toEqual([1, 2])
    // a failing update does not stop the others
    vi.spyOn(console, "error").mockImplementation(() => {})
    paintBatch(() => {
      throw new Error("boom")
    })
    paintBatch(() => applied.push(3))
    expect(applied).toEqual([1, 2, 3])
  })
})
