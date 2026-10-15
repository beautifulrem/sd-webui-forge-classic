import { describe, expect, it } from "vitest"
import { type Box, dragBox, LAYOUTS, MIN_SIZE, nudgeBox, quantize, snap, snapTargets } from "./geometry"

const LEFT: Box = { x: 0, y: 0, width: 0.5, height: 1 }
const CENTRE: Box = { x: 0.25, y: 0.25, width: 0.5, height: 0.5 }

describe("region geometry", () => {
  it("quantizes to the slider grid without float noise", () => {
    expect(quantize(0.1 + 0.2, 0.01)).toBe(0.3)
    expect(quantize(0.333, 0.01)).toBe(0.33)
    expect(quantize(0.4, 0)).toBe(0.4)
  })

  it("snaps to the nearest target within the threshold", () => {
    expect(snap(0.49, [0, 0.5, 1], 0.015)).toBe(0.5)
    expect(snap(0.45, [0, 0.5, 1], 0.015)).toBe(0.45)
    expect(snap(0.505, [0.5, 0.51], 0.015)).toBe(0.51)
  })

  it("moves a box inside the image, keeping its size", () => {
    expect(dragBox(CENTRE, "move", 0.1, -0.05, { step: 0.01 })).toEqual({ x: 0.35, y: 0.2, width: 0.5, height: 0.5 })
    expect(dragBox(CENTRE, "move", 0.9, 0.9, { step: 0.01 })).toEqual({ x: 0.5, y: 0.5, width: 0.5, height: 0.5 })
    expect(dragBox(CENTRE, "move", -1, -1, { step: 0.01 })).toEqual({ x: 0, y: 0, width: 0.5, height: 0.5 })
  })

  it("moves an edge onto a snap target", () => {
    const snaps = snapTargets([LEFT])
    // The left edge lands on region 1's right edge (0.5)
    expect(dragBox(CENTRE, "move", 0.24, 0, { step: 0.01, snaps, threshold: 0.015 }).x).toBe(0.5)
    // Snapping off: exactly where the pointer is
    expect(dragBox(CENTRE, "move", 0.24, 0, { step: 0.01, snaps, threshold: 0 }).x).toBe(0.49)
  })

  it("resizes the held edges within the image and the minimum size", () => {
    expect(dragBox(CENTRE, "se", 0.1, 0.1, { step: 0.01 })).toEqual({ x: 0.25, y: 0.25, width: 0.6, height: 0.6 })
    expect(dragBox(CENTRE, "nw", -0.1, -0.3, { step: 0.01 })).toEqual({ x: 0.15, y: 0, width: 0.6, height: 0.75 })
    expect(dragBox(CENTRE, "e", 1, 0, { step: 0.01 })).toEqual({ x: 0.25, y: 0.25, width: 0.75, height: 0.5 })
    expect(dragBox(CENTRE, "w", 1, 0, { step: 0.01 }).width).toBeCloseTo(MIN_SIZE)
    expect(dragBox(CENTRE, "n", 0, 0.1, { step: 0.01 })).toEqual({ x: 0.25, y: 0.35, width: 0.5, height: 0.4 })
  })

  it("draws a box past the image edge as clipped when it is resized", () => {
    const overflowing = { x: 0.8, y: 0, width: 0.5, height: 1 }
    expect(dragBox(overflowing, "w", -0.1, 0, { step: 0.01 })).toEqual({ x: 0.7, y: 0, width: 0.3, height: 1 })
  })

  it("nudges with the arrow keys", () => {
    expect(nudgeBox(CENTRE, "ArrowRight", 0.01, false, 0.01)).toEqual({ ...CENTRE, x: 0.26 })
    expect(nudgeBox(CENTRE, "ArrowUp", 0.1, false, 0.01)).toEqual({ ...CENTRE, y: 0.15 })
    expect(nudgeBox(CENTRE, "ArrowLeft", 0.01, true, 0.01)).toEqual({ ...CENTRE, width: 0.49 })
    expect(nudgeBox(CENTRE, "ArrowDown", 0.01, true, 0.01)).toEqual({ ...CENTRE, height: 0.51 })
  })

  it("offers layouts that tile the image", () => {
    for (const layout of LAYOUTS) {
      const area = layout.boxes.reduce((sum, box) => sum + box.width * box.height, 0)
      expect(area).toBeCloseTo(1)
      for (const box of layout.boxes) {
        expect(box.x + box.width).toBeLessThanOrEqual(1.000001)
        expect(box.y + box.height).toBeLessThanOrEqual(1.000001)
      }
    }
  })
})
