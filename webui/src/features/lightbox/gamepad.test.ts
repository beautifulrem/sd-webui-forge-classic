import { describe, expect, it } from "vitest"
import { padDirection } from "./gamepad"

const pad = (x: number, pressed: number[] = []) => ({ axes: [x, 0], buttons: Array.from({ length: 17 }, (_, i) => ({ pressed: pressed.includes(i) })) })

describe("lightbox gamepad", () => {
  it("reads the left stick past the dead zone and the d-pad", () => {
    expect(padDirection([null, pad(0.2)])).toBe(0)
    expect(padDirection([pad(-0.8)])).toBe(-1)
    expect(padDirection([null, pad(0.9)])).toBe(1)
    expect(padDirection([pad(0, [14])])).toBe(-1)
    expect(padDirection([pad(0, [15])])).toBe(1)
  })
})
