import { describe, expect, it } from "vitest"
import { applyStyleTexts, materializeStyles } from "./apply-styles"

describe("styles", () => {
  it("appends plain styles and wraps around {prompt}", () => {
    expect(applyStyleTexts("1girl", ["masterpiece, {prompt}, absurdres"])).toBe("masterpiece, 1girl, absurdres")
    expect(applyStyleTexts(" 1girl ", ["watercolor"])).toBe("1girl, watercolor")
    expect(applyStyleTexts("", ["watercolor"])).toBe("watercolor")
    expect(applyStyleTexts("cat", ["", null])).toBe("cat")
  })

  it("materializes the selected styles in order, prompt and negative", () => {
    const styles = [
      { name: "A", prompt: "best, {prompt}", negative_prompt: "worst" },
      { name: "B", prompt: "film grain", negative_prompt: "" },
    ]
    expect(materializeStyles({ prompt: "cat", negative: "blurry" }, ["A", "B", "missing"], styles)).toEqual({
      prompt: "best, cat, film grain",
      negative: "blurry, worst",
    })
  })
})
