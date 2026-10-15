import { describe, expect, it } from "vitest"
import { activeFilters, galleryQuery, localDayStart, NO_FILTERS } from "./filters"
import { aspectOf, justifiedNeighbor, justifiedRows, masonryColumns, rowOf } from "./layout"
import { EMPTY_SELECTION, ordered, prune, select, selectAll } from "./selection"

describe("justified layout", () => {
  it("fills each row to the container width", () => {
    const rows = justifiedRows([1, 1, 2, 1, 0.5], 400, 100, 10)
    expect(rows.map((row) => [row.start, row.end])).toEqual([
      [0, 3],
      [3, 5],
    ])
    const [first] = rows
    const width = first!.widths.reduce((a, b) => a + b, 0) + 10 * 2
    expect(width).toBeCloseTo(400)
    expect(first!.height).toBeCloseTo(380 / 4)
  })

  it("keeps the last row at the target height and full rows below it", () => {
    const rows = justifiedRows([1], 1000, 100, 8)
    expect(rows).toEqual([{ start: 0, end: 1, height: 100, widths: [100] }])
    const narrow = justifiedRows([0.2, 0.2, 3], 30, 100, 0)
    expect(narrow[0]).toMatchObject({ start: 0, end: 2, height: 75 })
    expect(narrow[1]!.height).toBe(10)
    expect(justifiedRows([1, 2], 0, 100, 8)).toEqual([])
  })

  it("reads aspect ratios safely", () => {
    expect(aspectOf(1024, 512)).toBe(2)
    expect(aspectOf(null, 10)).toBe(1)
    expect(aspectOf(10000, 10)).toBe(5)
    expect(aspectOf(10, 10000)).toBe(0.2)
  })

  it("finds the image above or below by horizontal position", () => {
    // Rows: [0 1 2] [3 4] [5], 10 px gaps.
    const rows = [
      { start: 0, end: 3, height: 100, widths: [100, 100, 180] },
      { start: 3, end: 5, height: 100, widths: [300, 90] },
      { start: 5, end: 6, height: 100, widths: [50] },
    ]
    expect(rowOf(rows, 4)).toBe(1)
    expect(rowOf(rows, 9)).toBe(-1)
    expect(justifiedNeighbor(rows, 0, 1, 10)).toBe(3)
    expect(justifiedNeighbor(rows, 2, 1, 10)).toBe(4)
    expect(justifiedNeighbor(rows, 4, -1, 10)).toBe(2)
    expect(justifiedNeighbor(rows, 4, 1, 10)).toBe(5) // past the short row's end: its last item
    expect(justifiedNeighbor(rows, 1, -1, 10)).toBeNull()
    expect(justifiedNeighbor(rows, 5, 1, 10)).toBeNull()
  })

  it("fits masonry columns", () => {
    expect(masonryColumns(1000, 200, 10)).toEqual({ count: 5, width: 192 })
    expect(masonryColumns(100, 300, 10).count).toBe(1)
  })
})

describe("gallery filters", () => {
  it("builds the query, dates as local day bounds", () => {
    expect(galleryQuery({ ...NO_FILTERS, sort: "newest" })).toEqual({ sort: "newest" })
    const query = galleryQuery({ q: "  fox ", category: "grids", model: "anima", from: "2026-01-02", to: "2026-01-02", starred: true, sort: "oldest" })
    expect(query).toMatchObject({ q: "fox", category: "grids", model: "anima", starred: true, sort: "oldest" })
    expect(new Date(query.since!).getDate()).toBe(2)
    expect(new Date(query.until!).getTime() - new Date(query.since!).getTime()).toBeGreaterThanOrEqual(23 * 3600_000)
  })

  it("ignores malformed dates and counts active filters", () => {
    expect(localDayStart("yesterday")).toBeNull()
    expect(localDayStart(null)).toBeNull()
    expect(activeFilters({ ...NO_FILTERS, sort: "name" })).toBe(0)
    expect(activeFilters({ ...NO_FILTERS, sort: "name", q: "x", from: "2026-01-01", starred: true })).toBe(3)
  })
})

describe("selection", () => {
  const order = ["a", "b", "c", "d"]

  it("toggles, extends ranges and replaces", () => {
    let s = select(EMPTY_SELECTION, order, "b", "toggle")
    s = select(s, order, "d", "range")
    expect(ordered(s, order)).toEqual(["b", "c", "d"])
    s = select(s, order, "c", "toggle")
    expect(ordered(s, order)).toEqual(["b", "d"])
    expect(ordered(select(s, order, "a", "only"), order)).toEqual(["a"])
  })

  it("selects all and prunes vanished ids", () => {
    const all = selectAll(order)
    expect(all.ids.size).toBe(4)
    const pruned = prune(all, ["a", "c"])
    expect(ordered(pruned, ["a", "c"])).toEqual(["a", "c"])
    expect(prune(pruned, ["a", "c"])).toBe(pruned)
  })
})
