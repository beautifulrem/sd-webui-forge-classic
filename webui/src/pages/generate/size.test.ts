import { describe, expect, it } from "vitest"
import { aspectSize, matchAspect, parseSize } from "./size"

describe("aspectSize", () => {
  it("keeps the pixel budget on the 64 grid", () => {
    expect(aspectSize([1, 1], 1024 * 1024)).toEqual({ width: 1024, height: 1024 })
    expect(aspectSize([3, 2], 1024 * 1024)).toEqual({ width: 1280, height: 832 })
    expect(aspectSize([9, 16], 1024 * 1024)).toEqual({ width: 768, height: 1344 })
  })

  it("clamps to the bounds", () => {
    expect(aspectSize([21, 9], 2048 * 2048, { maximum: 2048 })).toEqual({ width: 2048, height: 1344 })
  })
})

describe("matchAspect", () => {
  it("recognises common ratios within 3 %", () => {
    expect(matchAspect({ width: 1216, height: 832 })).toEqual([3, 2])
    expect(matchAspect({ width: 1000, height: 333 })).toBeNull()
  })
})

describe("parseSize", () => {
  it.each([
    ["1024x768", { width: 1024, height: 768 }],
    [" 832 × 1216 ", { width: 832, height: 1216 }],
    ["640*480", { width: 640, height: 480 }],
    ["512, 512", { width: 512, height: 512 }],
    ["1024 768", { width: 1024, height: 768 }],
  ])("%s", (text, size) => expect(parseSize(text)).toEqual(size))

  it("rejects other text", () => {
    expect(parseSize("1girl, 1024x768 wallpaper")).toBeNull()
    expect(parseSize("16:9")).toBeNull()
  })
})
