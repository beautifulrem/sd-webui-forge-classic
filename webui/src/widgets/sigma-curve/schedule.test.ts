import { describe, expect, it } from "vitest"
import fixture from "./schedule.fixture.json"
import { linspace, makeSchedule, pyRound, type ScheduleParams, summarize } from "./schedule"

type Case = { steps: number; params: ScheduleParams; values: number[] }

describe("Detail Daemon schedule", () => {
  it("matches the extension's make_schedule for every fixture case", () => {
    // schedule.fixture.json is written from the Python function (extensions-builtin/sd-webui-detail-daemon/tests/schedule_cases.py)
    for (const { steps, params, values } of fixture as Case[]) {
      const got = makeSchedule(steps, params)
      expect(got, JSON.stringify(params)).toHaveLength(values.length)
      got.forEach((v, i) => expect(v, `${JSON.stringify(params)} step ${i}`).toBeCloseTo(values[i]!, 12))
    }
  })

  it("rounds halves to even like Python", () => {
    expect([0.5, 1.5, 2.5, 3.5, -2.5, 2.4, 2.6].map(pyRound)).toEqual([0, 2, 2, 4, -2, 2, 3])
  })

  it("spaces values like numpy.linspace", () => {
    expect(linspace(0, 1, 5)).toEqual([0, 0.25, 0.5, 0.75, 1])
    expect(linspace(1, 0, 1)).toEqual([1])
    expect(linspace(0, 1, 0)).toEqual([])
  })

  it("summarizes the mean, peak and bias position", () => {
    const params = { ...(fixture as Case[])[0]!.params, amount: -0.5 }
    const { mean, peak, mid } = summarize(makeSchedule(50, params), params)
    expect(mean).toBeLessThan(0)
    expect(peak).toBeCloseTo(0.5, 6)
    expect(mid).toBeCloseTo(0.5, 12)
  })
})
