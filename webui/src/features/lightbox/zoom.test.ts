import { describe, expect, it } from "vitest"
import { clampPan, fitScale, initialView, pinchView, wheelFactor, zoomAt } from "./zoom"

const image = { width: 2000, height: 1000 }
const viewport = { width: 1000, height: 800 }

describe("lightbox zoom", () => {
  it("fits without enlarging", () => {
    expect(fitScale(image, viewport)).toBe(0.5)
    expect(fitScale({ width: 100, height: 100 }, viewport)).toBe(1)
  })

  it("starts at actual size when zoomed by default, else fitted", () => {
    expect(initialView(image, viewport, true)).toEqual({ scale: 1, x: 0, y: 0 })
    expect(initialView(image, viewport, false)).toEqual({ scale: 0.5, x: 0, y: 0 })
  })

  it("zooms around the cursor", () => {
    const view = zoomAt({ scale: 1, x: 0, y: 0 }, 2, { x: 100, y: 50 })
    expect(view).toEqual({ scale: 2, x: -100, y: -50 })
    // The image point under the cursor stays there: (100 - x) / scale is unchanged.
    expect((100 - view.x) / view.scale).toBe(100)
  })

  it("keeps the image over the viewport while panning", () => {
    expect(clampPan({ scale: 1, x: 900, y: -300 }, image, viewport)).toEqual({ scale: 1, x: 500, y: -100 })
    expect(clampPan({ scale: 0.25, x: 40, y: 40 }, image, viewport)).toEqual({ scale: 0.25, x: 0, y: 0 })
  })

  it("turns wheel deltas into zoom factors", () => {
    expect(wheelFactor(-100)).toBeGreaterThan(1)
    expect(wheelFactor(100)).toBeLessThan(1)
    expect(wheelFactor(-3, 1)).toBeCloseTo(wheelFactor(-48))
  })

  it("pinches around the fingers' midpoint and pans with it", () => {
    const start = { scale: 1, x: 0, y: 0 }
    // Fingers spread from 100 px to 200 px apart around (50, 0): the image point under the midpoint stays.
    const spread = pinchView(start, [{ x: 0, y: 0 }, { x: 100, y: 0 }], [{ x: -50, y: 0 }, { x: 150, y: 0 }])
    expect(spread).toEqual({ scale: 2, x: -50, y: 0 })
    expect((50 - spread.x) / spread.scale).toBe(50)
    // Both fingers move together: a pan only.
    expect(pinchView(start, [{ x: 0, y: 0 }, { x: 100, y: 0 }], [{ x: 10, y: 20 }, { x: 110, y: 20 }])).toEqual({ scale: 1, x: 10, y: 20 })
  })
})
