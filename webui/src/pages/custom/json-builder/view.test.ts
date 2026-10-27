import { describe, expect, it } from "vitest"
import { clampAxis, clampView, fitStage, MAX_ZOOM, MIN_ZOOM, RESET_VIEW, zoomAbout } from "./view"

describe("canvas view", () => {
  it("fits the stage at the draft's aspect ratio", () => {
    expect(fitStage(800, 600, 1)).toEqual({ width: 600, height: 600 })
    expect(fitStage(800, 600, 16 / 9)).toEqual({ width: 800, height: 450 })
    expect(fitStage(0, 0, 1)).toEqual({ width: 1, height: 1 })
  })

  it("pans a larger stage up to its edges, a smaller one within the area", () => {
    expect(clampAxis(500, 600, 1200)).toBe(300)
    expect(clampAxis(-500, 600, 1200)).toBe(-300)
    expect(clampAxis(500, 600, 400)).toBe(100)
    expect(clampView({ scale: 2, panX: 900, panY: -900 }, { width: 600, height: 600 }, { width: 600, height: 300 })).toEqual({ scale: 2, panX: 300, panY: 0 })
  })

  it("zooms about a point that stays under the pointer", () => {
    const view = zoomAbout(RESET_VIEW, 100, -50, 2)
    expect(view).toEqual({ scale: 2, panX: -100, panY: 50 })
    // the stage point under (100, -50) before the zoom is still there: (x - pan) / scale
    expect((100 - view.panX) / view.scale).toBe(100)
    expect(zoomAbout({ scale: MAX_ZOOM, panX: 0, panY: 0 }, 10, 10, 2)).toEqual({ scale: MAX_ZOOM, panX: 0, panY: 0 })
    expect(zoomAbout(RESET_VIEW, 0, 0, 0.01).scale).toBe(MIN_ZOOM)
  })
})
