import { describe, expect, it } from "vitest"
import { clampPosition, keyPosition } from "./position"

describe("compare slider position", () => {
  it("clamps and rounds", () => {
    expect(clampPosition(-1)).toBe(0)
    expect(clampPosition(2)).toBe(1)
    expect(clampPosition(0.12345)).toBe(0.123)
    expect(clampPosition(Number.NaN)).toBe(0.5)
  })

  it("moves with the keyboard", () => {
    expect(keyPosition(0.5, "ArrowRight", false)).toBe(0.51)
    expect(keyPosition(0.5, "ArrowLeft", true)).toBe(0.4)
    expect(keyPosition(0.95, "PageUp", false)).toBe(1)
    expect(keyPosition(0.3, "Home", false)).toBe(0)
    expect(keyPosition(0.3, "End", false)).toBe(1)
    expect(keyPosition(0.3, "a", false)).toBeNull()
  })
})
