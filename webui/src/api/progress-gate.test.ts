import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

describe("createProgressGate", () => {
  let clock = 0
  beforeEach(() => {
    vi.resetModules()
    vi.useFakeTimers()
    clock = 0
    // hidden page: paint batches apply at once, so this tests only the gate
    vi.spyOn(document, "visibilityState", "get").mockReturnValue("hidden")
  })
  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  async function setup() {
    const { createProgressGate, MAX_HOLD_MS } = await import("./progress-gate")
    const applied: number[] = []
    const gate = createProgressGate<{ seq: number; step: number }>(
      (e) => applied.push(e.step),
      () => clock,
    )
    const at = (ms: number) => {
      vi.advanceTimersByTime(ms - clock)
      clock = ms
    }
    return { gate, applied, at, MAX_HOLD_MS }
  }

  it("applies progress at once until previews arrive for every step", async () => {
    const { gate, applied, at } = await setup()
    gate.progress({ seq: 1, step: 0 })
    expect(applied).toEqual([0])
    at(1000)
    gate.frame({ jobSeq: 1, step: 0 })
    gate.progress({ seq: 1, step: 1 }) // one frame: no step interval known yet
    expect(applied).toEqual([0, 1])
  })

  it("holds a step's progress for the next preview frame while frames come every step", async () => {
    const { gate, applied, at } = await setup()
    gate.frame({ jobSeq: 1, step: 0 })
    at(3000)
    gate.frame({ jobSeq: 1, step: 1 }) // interval 3 s
    at(3060)
    gate.progress({ seq: 1, step: 2 }) // the sampler started step 2: waits for its frame
    expect(applied).toEqual([])
    at(6000)
    gate.frame({ jobSeq: 1, step: 2 })
    expect(applied).toEqual([2])
    // only the newest held event is drawn
    at(6050)
    gate.progress({ seq: 1, step: 3 })
    gate.progress({ seq: 1, step: 3 })
    at(9000)
    gate.frame({ jobSeq: 1, step: 3 })
    expect(applied).toEqual([2, 3])
  })

  it("draws a held progress on its own when the next frame is late", async () => {
    const { gate, applied, at } = await setup()
    gate.frame({ jobSeq: 1, step: 0 })
    at(1000)
    gate.frame({ jobSeq: 1, step: 1 })
    at(1050)
    gate.progress({ seq: 1, step: 2 })
    at(2999)
    expect(applied).toEqual([])
    at(3001) // two intervals after the last frame
    expect(applied).toEqual([2])
    // and later events (still no frame) are drawn at once
    gate.progress({ seq: 1, step: 3 })
    expect(applied).toEqual([2, 3])
  })

  it("does not hold for previews every few steps, another job or after a flush", async () => {
    const { gate, applied, at } = await setup()
    gate.frame({ jobSeq: 1, step: 0 })
    at(1000)
    gate.frame({ jobSeq: 1, step: 4 }) // every 4 steps (or throttled): not consecutive
    gate.progress({ seq: 1, step: 5 })
    expect(applied).toEqual([5])
    at(2000)
    gate.frame({ jobSeq: 1, step: 5 })
    gate.progress({ seq: 2, step: 0 }) // another job's progress
    expect(applied).toEqual([5, 0])
    at(3000)
    gate.frame({ jobSeq: 1, step: 6 })
    gate.progress({ seq: 1, step: 7 })
    expect(applied).toEqual([5, 0])
    gate.flush() // the job ended: the held step is applied first
    expect(applied).toEqual([5, 0, 7])
    gate.progress({ seq: 1, step: 8 })
    expect(applied).toEqual([5, 0, 7, 8])
  })

  it("caps the hold", async () => {
    const { gate, applied, at, MAX_HOLD_MS } = await setup()
    gate.frame({ jobSeq: 1, step: 0 })
    at(60_000)
    gate.frame({ jobSeq: 1, step: 1 }) // one-minute steps
    at(60_010)
    gate.progress({ seq: 1, step: 2 })
    at(60_000 + MAX_HOLD_MS - 1)
    expect(applied).toEqual([])
    at(60_000 + MAX_HOLD_MS + 1)
    expect(applied).toEqual([2])
  })
})
