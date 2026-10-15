import { describe, expect, it } from "vitest"
import { boundsOf, fromThumbs, normalizeRange, setEnd } from "./range"

const unit = boundsOf({ minimum: 0, maximum: 1, step: 0.05 })

describe("range-slider values", () => {
  it("reads the bounds from the props, with defaults for bad ones", () => {
    expect(boundsOf({})).toEqual({ minimum: 0, maximum: 1, step: 0.01 })
    expect(boundsOf({ minimum: 0.5, maximum: 2, step: 0.25 })).toEqual({ minimum: 0.5, maximum: 2, step: 0.25 })
    expect(boundsOf({ minimum: 3, maximum: 1, step: -1 })).toEqual({ minimum: 0, maximum: 1, step: 0.01 })
    expect(boundsOf({ minimum: "0", maximum: Number.NaN })).toEqual({ minimum: 0, maximum: 1, step: 0.01 })
  })

  it("keeps a pair ordered and inside the bounds, as the server validates it", () => {
    expect(normalizeRange([0.15, 0.85], unit)).toEqual([0.15, 0.85])
    expect(normalizeRange([0.9, 0.1], unit)).toEqual([0.1, 0.9])
    expect(normalizeRange([-1, 2], unit)).toEqual([0, 1])
    for (const bad of [null, undefined, 0.5, [0.5], [0.1, "0.2"], [0.1, Number.NaN], { 0: 0.1, 1: 0.2 }]) {
      expect(normalizeRange(bad, unit)).toEqual([0, 1])
    }
  })

  it("never lets the typed ends cross", () => {
    expect(setEnd([0.2, 0.6], 0, 0.4, unit)).toEqual([0.4, 0.6])
    expect(setEnd([0.2, 0.6], 0, 0.9, unit)).toEqual([0.6, 0.6])
    expect(setEnd([0.2, 0.6], 1, 0.1, unit)).toEqual([0.2, 0.2])
    expect(setEnd([0.2, 0.6], 1, 7, unit)).toEqual([0.2, 1])
  })

  it("snaps the thumbs to the step grid from the minimum", () => {
    expect(fromThumbs([0.123, 0.876], unit)).toEqual([0.1, 0.9])
    expect(fromThumbs([0.7, 0.3], unit)).toEqual([0.3, 0.7])
    expect(fromThumbs([300, 1000], boundsOf({ minimum: 256, maximum: 2048, step: 64 }))).toEqual([320, 1024])
  })
})
