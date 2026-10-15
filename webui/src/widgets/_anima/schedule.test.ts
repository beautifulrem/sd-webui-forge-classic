import { describe, expect, it } from "vitest"
import fixture from "./schedule.fixture.json"
import { curvePath, flowSigma, stageWindow, windowCurveFactor, type StageKind } from "./schedule"

describe("stageWindow", () => {
  it.each(fixture.windows)("matches Python's stage_window($stage, $shift)", ({ stage, shift, window }) => {
    const got = stageWindow(stage as StageKind, shift)
    expect(got.start).toBeCloseTo(window[0]!, 12)
    expect(got.end).toBeCloseTo(window[1]!, 12)
    expect(got.peak).toBeCloseTo(window[2]!, 12)
  })

  it("reads a missing Shift as 3", () => {
    expect(stageWindow("style", Number.NaN)).toEqual(stageWindow("style", 3))
  })
})

describe("windowCurveFactor", () => {
  it.each(fixture.curves)("matches Python's window_curve_factor ($shape, floor $floor, $start-$end peak $peak)", ({ shape, floor, start, end, peak, values }) => {
    fixture.points.forEach((x, i) => expect(windowCurveFactor(x, start, end, peak, shape, floor)).toBeCloseTo(values[i]!, 12))
  })
})

describe("flowSigma and paths", () => {
  it("runs from 1 to 0 and bends with the shift", () => {
    expect(flowSigma(0, 3)).toBe(1)
    expect(flowSigma(1, 3)).toBe(0)
    expect(flowSigma(0.5, 3)).toBeCloseTo(0.75, 12)
    expect(flowSigma(0.5, 1)).toBeCloseTo(0.5, 12)
  })

  it("draws a curve inside its box", () => {
    const path = curvePath(() => 1, 100, 20, 4)
    expect(path).toBe("M0.00,0.00L25.00,0.00L50.00,0.00L75.00,0.00L100.00,0.00")
  })
})
