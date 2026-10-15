import { describe, expect, it } from "vitest"
import { draftFrom, patchFrom } from "./metadata-draft"

const ALL = ["description", "sd version", "activation text", "preferred weight", "negative text", "tags", "notes"]

describe("metadata draft", () => {
  it("reads the sidecar, tolerating odd values", () => {
    const draft = draftFrom({ description: null, user_metadata: { "preferred weight": "0.7", tags: ["a", 3], "sd version": "xl", notes: 5 } })
    expect(draft).toMatchObject({ weight: 0.7, tags: "a, 3", sdVersion: "xl", notes: "", activation: "" })
    expect(draftFrom({ description: null, user_metadata: { "preferred weight": "heavy", tags: "x, y" } })).toMatchObject({ weight: 0, tags: "x, y", sdVersion: "Unknown" })
  })

  it("removes empty keys: weight 0, Unknown, blank text and tags", () => {
    const patch = patchFrom({ description: "  ", sdVersion: "Unknown", activation: " ", weight: 0, negative: "", tags: " , ,", notes: "" }, ALL)
    expect(patch).toEqual({ description: null, "sd version": null, "activation text": null, "preferred weight": null, "negative text": null, tags: null, notes: null })
  })

  it("writes trimmed text, split tags and only the kind's fields", () => {
    const draft = { description: "soft look\n", sdVersion: "anima", activation: " wc, soft ", weight: 0.65, negative: " blurry ", tags: "paint, style, ,", notes: "n" }
    expect(patchFrom(draft, ALL)).toEqual({
      description: "soft look\n",
      "sd version": "anima",
      "activation text": "wc, soft",
      "preferred weight": 0.65,
      "negative text": "blurry",
      tags: ["paint", "style"],
      notes: "n",
    })
    expect(patchFrom(draft, ["description", "tags"])).toEqual({ description: "soft look\n", tags: ["paint", "style"] })
  })
})
