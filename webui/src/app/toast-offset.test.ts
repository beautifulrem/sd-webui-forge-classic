import { describe, expect, it } from "vitest"
import { GAP, MOBILE_OFFSET, OFFSET, toastBottomOffset } from "./toast-offset"

const box = (left: number, top: number, right: number, bottom: number) => ({ left, top, right, bottom })

describe("toastBottomOffset", () => {
  it("keeps sonner's offsets when nothing is pinned to the bottom", () => {
    expect(toastBottomOffset([], 1280, 720)).toBe(OFFSET)
    expect(toastBottomOffset([], 420, 860)).toBe(MOBILE_OFFSET)
  })

  it("lifts the toasts above a phone's bottom navigation and sticky Generate bar", () => {
    const nav = box(0, 804, 420, 860)
    const generateBar = box(0, 740, 420, 804)
    expect(toastBottomOffset([nav, generateBar], 420, 860)).toBe(860 - 740 + GAP)
  })

  it("lifts the toasts above the canvas toolbars under their column on a desktop", () => {
    const layers = box(838, 660, 1040, 700) // reaches into the toast column (right 380px)
    expect(toastBottomOffset([layers], 1280, 720)).toBe(720 - 660 + GAP)
  })

  it("ignores controls outside the toast column, hidden ones and ones in the upper half", () => {
    const zoom = box(494, 660, 730, 700) // bottom-left of the stage
    const hidden = box(0, 0, 0, 0)
    const generateBar = box(68, 60, 460, 100)
    const offscreen = box(900, 760, 1200, 800)
    expect(toastBottomOffset([zoom, hidden, generateBar, offscreen], 1280, 720)).toBe(OFFSET)
  })
})
