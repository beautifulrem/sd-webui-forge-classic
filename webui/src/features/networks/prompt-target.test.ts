import { afterEach, describe, expect, it } from "vitest"
import { getFormStore } from "@/ui/form-store"
import { planInsertion, readPrompts, writePrompts } from "./prompt-target"

const options = { negativeAllowed: true, target: "positive" as const, separator: " " }

describe("inserting cards into prompts", () => {
  afterEach(() => getFormStore("txt2img").setState({ values: {} }))

  it("toggles text in the target prompt", () => {
    const item = { prompt: "<lora:ink:0.8>", negative_prompt: null }
    expect(planInsertion({ positive: "1girl", negative: "" }, item, options)).toEqual({ changes: { positive: "1girl <lora:ink:0.8>" }, added: true })
    expect(planInsertion({ positive: "1girl <lora:ink:1.2>", negative: "" }, item, options)).toEqual({ changes: { positive: "1girl" }, added: false })
    expect(planInsertion({ positive: "", negative: "lowres" }, { prompt: "easynegative", negative_prompt: null }, { ...options, target: "negative" })).toEqual({
      changes: { negative: "lowres easynegative" },
      added: true,
    })
  })

  it("keeps kinds without negative use in the positive prompt", () => {
    const plan = planInsertion({ positive: "", negative: "" }, { prompt: "<hypernet:x:1>", negative_prompt: null }, { ...options, negativeAllowed: false, target: "negative" })
    expect(plan?.changes).toEqual({ positive: "<hypernet:x:1>" })
  })

  it("adds and removes a card's negative text together with its prompt", () => {
    const item = { prompt: "<lora:wc:1> watercolor", negative_prompt: "(blurry:1.2)" }
    const added = planInsertion({ positive: "a", negative: "lowres" }, item, { ...options, target: "negative" })
    expect(added).toEqual({ changes: { positive: "a <lora:wc:1> watercolor", negative: "lowres (blurry:1.2)" }, added: true })
    const removed = planInsertion({ positive: added!.changes.positive!, negative: added!.changes.negative! }, item, options)
    expect(removed).toEqual({ changes: { positive: "a", negative: "lowres" }, added: false })
    // the negative text was already there: it stays, only the prompt changes
    expect(planInsertion({ positive: "", negative: "(blurry:1.2)" }, item, options)?.changes).toEqual({ positive: "<lora:wc:1> watercolor" })
  })

  it("ignores cards without text", () => {
    expect(planInsertion({ positive: "", negative: "" }, { prompt: "  ", negative_prompt: null }, options)).toBeNull()
  })

  it("reads and writes the form's core prompts, also before its schema loaded", () => {
    writePrompts("txt2img", { positive: "a cat", negative: "lowres" })
    expect(readPrompts("txt2img")).toEqual({ positive: "a cat", negative: "lowres" })
    expect(getFormStore("txt2img").getState().values.core).toEqual({ prompt: "a cat", negative_prompt: "lowres" })
  })
})
