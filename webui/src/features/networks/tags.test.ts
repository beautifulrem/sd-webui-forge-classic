import { describe, expect, it } from "vitest"
import { escapeTag, hasTag, randomPrompt, tagWeight, toggleTag } from "./tags"

describe("dataset tags", () => {
  it("escapes brackets", () => {
    expect(escapeTag("artist (style) [x]")).toBe("artist \\(style\\) \\[x\\]")
  })

  it("picks tags by their frequency", () => {
    const tags = [
      { tag: "1girl", count: 10 },
      { tag: "solo", count: 5 },
      { tag: "hat (red)", count: 1 },
    ]
    expect(randomPrompt(tags, () => 0)).toBe("1girl, hat \\(red\\), solo")
    expect(randomPrompt(tags, () => 0.6)).toBe("1girl")
    expect(randomPrompt([], () => 0)).toBe("")
  })

  it("toggles a tag in comma-separated text", () => {
    expect(toggleTag("", "a")).toBe("a")
    expect(toggleTag("a, b", "c")).toBe("a, b, c")
    expect(toggleTag("a, b ,c", "b")).toBe("a, c")
    expect(hasTag("a, b", "b")).toBe(true)
    expect(hasTag("a, bc", "b")).toBe(false)
  })

  it("weights chips", () => {
    expect(tagWeight(25, 100)).toBe(0.5)
    expect(tagWeight(5, 0)).toBe(0)
  })
})
