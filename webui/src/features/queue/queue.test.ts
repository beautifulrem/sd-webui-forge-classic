import { describe, expect, it } from "vitest"
import { jobEntries } from "./edit-session"
import { durationOf, formatDuration } from "./queries"
import { dropTarget, moveIndex, planMove, priorityLabel } from "./reorder"

const q = (id: string, priority = 0) => ({ id, priority })

describe("queue reorder planning", () => {
  it("moves within a priority group, next to a neighbour", () => {
    expect(moveIndex(["a", "b", "c"], 0, 2)).toEqual(["b", "c", "a"])
    expect(planMove([q("a"), q("b"), q("c")], 2, 0)).toEqual({ id: "c", order: ["c", "a", "b"], priority: null, after: null, before: "a" })
    expect(planMove([q("a"), q("b"), q("c")], 0, 1)).toMatchObject({ after: "b", before: "c" })
    expect(planMove([q("a"), q("b")], 1, 1)).toBeNull()
  })

  it("sends one neighbour whatever the length of the queue", () => {
    const jobs = Array.from({ length: 600 }, (_, index) => q(`j${index}`))
    const plan = planMove(jobs, 599, 0)!
    expect(plan).toMatchObject({ id: "j599", after: null, before: "j0", priority: null })
    expect(plan.order).toHaveLength(600)
    expect(planMove(jobs, 0, 599)).toMatchObject({ id: "j0", after: "j599", before: null })
  })

  it("takes the priority of the group it is dropped into", () => {
    const jobs = [q("hi1", 10), q("hi2", 10), q("n1"), q("n2"), q("lo", -10)]
    // Normal job dragged between the two high ones.
    expect(planMove(jobs, 3, 1)).toEqual({ id: "n2", order: ["hi1", "n2", "hi2", "n1", "lo"], priority: 10, after: "hi1", before: "hi2" })
    // High job dragged to the very end joins the low group.
    expect(planMove(jobs, 0, 4)).toMatchObject({ priority: -10, after: "lo", before: null })
    // Dropped between two high jobs, a normal one becomes high.
    expect(planMove(jobs, 2, 1)).toMatchObject({ order: ["hi1", "n1", "hi2", "n2", "lo"], priority: 10 })
    // A high job dropped among normal ones becomes normal (the list stays in run order).
    expect(planMove(jobs, 1, 3)).toMatchObject({ order: ["hi1", "n1", "n2", "hi2", "lo"], priority: 0, after: "n2", before: null })
    expect(planMove(jobs, 4, 4)).toBeNull()
  })

  it("plans a drop by id against the latest queue", () => {
    const arranged = [q("a"), q("b"), q("x"), q("c")] // x was dragged between b and c
    // a started meanwhile: x still lands after b
    expect(dropTarget([q("b"), q("c"), q("x")], arranged, "x")).toEqual({ from: 2, to: 1 })
    // b started too: after the nearest job above that is still queued (none: the top)
    expect(dropTarget([q("c"), q("x")], arranged, "x")).toEqual({ from: 1, to: 0 })
    // x itself started: nothing to move
    expect(dropTarget([q("b"), q("c")], arranged, "x")).toBeNull()
  })

  it("labels priorities", () => {
    expect(priorityLabel(0)).toBeNull()
    expect(priorityLabel(10)).toBe("High")
    expect(priorityLabel(3)).toBe("High")
    expect(priorityLabel(-1)).toBe("Low")
  })
})

describe("queue helpers", () => {
  it("measures and formats durations", () => {
    expect(durationOf({ started_at: "2026-01-01T00:00:00Z", finished_at: "2026-01-01T00:00:04.5Z" })).toBe(4.5)
    expect(durationOf({ started_at: null, finished_at: "2026-01-01T00:00:04Z" })).toBeNull()
    expect(formatDuration(4.5)).toBe("4.5 s")
    expect(formatDuration(42)).toBe("42 s")
    expect(formatDuration(185)).toBe("3 min 05 s")
    expect(formatDuration(3720)).toBe("1 h 02 min")
  })

  it("turns a job into one paste entry per owner", () => {
    expect(jobEntries({ core: { prompt: "x" }, scripts: { seed: { seed: 3 }, broken: 5 as unknown as Record<string, unknown> } })).toEqual([
      { owner: "core", values: { prompt: "x" } },
      { owner: "seed", values: { seed: 3 } },
    ])
  })
})
