import { describe, expect, it } from "vitest"
import { pairStats, pairsNeeded, repeatDecision, tQuantile, T95_ONE_SIDED } from "./stats.mjs"

describe("pairStats", () => {
  it("gives the mean with its two-sided interval and one-sided bounds", () => {
    const stats = pairStats([1, 2, 3, 4, 5])
    // mean 3, sd 1.581, se 0.707; t(4) two-sided 2.78, one-sided 2.13
    expect(stats).toMatchObject({ n: 5, mean: 3, ci95: 1.97, upper95: 4.51, lower95: 1.49, median: 3 })
  })

  it("has no bounds for fewer than two pairs", () => {
    expect(pairStats([])).toMatchObject({ n: 0, upper95: null, lower95: null })
    expect(pairStats([2])).toMatchObject({ n: 1, mean: 2, upper95: null, lower95: null })
  })

  it("uses the normal expansion beyond the table", () => {
    expect(tQuantile(T95_ONE_SIDED, 1.645, 5)).toBe(2.02)
    expect(tQuantile(T95_ONE_SIDED, 1.645, 200)).toBeCloseTo(1.653, 2)
  })
})

describe("repeatDecision (adaptive interleaved runs)", () => {
  const decide = (jobs, stats, { minJobs = 4, maxJobs = 20, target = 1 } = {}) => repeatDecision({ jobs, minJobs, maxJobs, target, stats })

  it("runs at least the minimum number of jobs", () => {
    expect(decide(2, { n: 26, upper95: 0.5, lower95: -1 })).toMatchObject({ stop: false })
  })

  it("stops once the upper bound is within the budget", () => {
    expect(decide(4, { n: 55, upper95: 0.98, lower95: -0.5 })).toMatchObject({ stop: true, verdict: "within" })
  })

  it("adds jobs while the bound straddles the budget, until the cap", () => {
    // the third review's run: 4 jobs, 55 pairs, +0.41 % with an upper bound of 1.13 %
    expect(decide(4, { n: 55, mean: 0.41, upper95: 1.13, lower95: -0.31 })).toMatchObject({ stop: false })
    expect(decide(20, { n: 260, upper95: 1.05, lower95: 0.1 })).toMatchObject({ stop: true, verdict: "inconclusive" })
  })

  it("stops when the overhead is shown to be over the budget", () => {
    expect(decide(5, { n: 60, upper95: 6.1, lower95: 2.4 })).toMatchObject({ stop: true, verdict: "over" })
  })

  it("runs exactly --repeat jobs when no cap above it is given", () => {
    expect(decide(9, { n: 110, upper95: 1.5, lower95: -0.2 }, { minJobs: 10, maxJobs: 10 })).toMatchObject({ stop: false })
    expect(decide(10, { n: 120, upper95: 1.5, lower95: -0.2 }, { minJobs: 10, maxJobs: 10 })).toMatchObject({ stop: true, verdict: "inconclusive" })
  })
})

describe("pairsNeeded", () => {
  it("grows with the scatter and with an overhead near the budget", () => {
    // the idle machine's 512x512 runs: pair SD 6-7.5 %
    expect(pairsNeeded({ sd: 7, expected: 0 })).toBe(133)
    expect(pairsNeeded({ sd: 7, expected: 0.3 })).toBe(271)
    expect(pairsNeeded({ sd: 3.2, expected: 0.4 })).toBe(77)
    expect(pairsNeeded({ sd: 3, expected: 1.2 })).toBeNull()
  })
})
