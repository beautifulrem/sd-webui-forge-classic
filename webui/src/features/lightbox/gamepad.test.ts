import { describe, expect, it } from "vitest"
import { padDirection, wheelDirection } from "./gamepad"

const pad = (x: number, pressed: number[] = []) => ({ axes: [x, 0], buttons: Array.from({ length: 17 }, (_, i) => ({ pressed: pressed.includes(i) })) })

describe("lightbox gamepad", () => {
  it("reads the left stick past the dead zone and the d-pad", () => {
    expect(padDirection([null, pad(0.2)])).toBe(0)
    expect(padDirection([pad(-0.8)])).toBe(-1)
    expect(padDirection([null, pad(0.9)])).toBe(1)
    expect(padDirection([pad(0, [14])])).toBe(-1)
    expect(padDirection([pad(0, [15])])).toBe(1)
  })

  it("turns the page on a mostly horizontal wheel only", () => {
    expect(wheelDirection({ deltaX: -40, deltaY: 3 })).toBe(-1)
    expect(wheelDirection({ deltaX: 1, deltaY: 0 })).toBe(1)
    expect(wheelDirection({ deltaX: 0.2, deltaY: 0 })).toBe(0)
    expect(wheelDirection({ deltaX: 10, deltaY: 30 })).toBe(0) // vertical: zoom
    expect(wheelDirection({ deltaX: 50, deltaY: 0, ctrlKey: true })).toBe(0) // pinch
  })
})
