import { describe, expect, it } from "vitest"
import { appendToPrompt, isInPrompt, parseTag, removeFromPrompt, togglePromptText } from "./prompt-insert"

describe("network card text in prompts", () => {
  it("parses a leading network tag", () => {
    expect(parseTag("<lora:wc_style:0.65> watercolor, soft")).toEqual({ type: "lora", name: "wc_style", rest: " watercolor, soft" })
    expect(parseTag("<LoRA:a b:1:0.5>")).toEqual({ type: "lora", name: "a b", rest: "" })
    expect(parseTag("easynegative")).toBeNull()
  })

  it("appends with the separator, never into an empty prompt", () => {
    expect(appendToPrompt("", "<lora:a:1>", " ")).toBe("<lora:a:1>")
    expect(appendToPrompt("  ", "<lora:a:1>", " ")).toBe("<lora:a:1>")
    expect(appendToPrompt("1girl", "<lora:a:1>", " ")).toBe("1girl <lora:a:1>")
    expect(appendToPrompt("1girl\n", "<lora:a:1>", " ")).toBe("1girl\n<lora:a:1>")
    expect(appendToPrompt("1girl, ", "x", ", ")).toBe("1girl, x")
    expect(appendToPrompt("1girl,", "x", ", ")).toBe("1girl, x")
    expect(appendToPrompt("1girl", "x", ", ")).toBe("1girl, x")
  })

  it("toggles a LoRA tag whatever its weight, with its activation text", () => {
    const card = "<lora:wc:0.65> watercolor, soft"
    const added = togglePromptText("1girl, smile", card, " ")
    expect(added).toEqual({ value: "1girl, smile <lora:wc:0.65> watercolor, soft", added: true })
    // the user changed the weight meanwhile
    const edited = added.value.replace("0.65", "1.2")
    expect(togglePromptText(edited, card, " ")).toEqual({ value: "1girl, smile", added: false })
    // tags of other networks stay
    expect(removeFromPrompt("<lora:wc2:1> <lora:wc:1> x", "<lora:wc:1>", " ")).toBe("<lora:wc2:1> x")
    expect(removeFromPrompt("<lora:wc:1>, a", "<lora:wc:1>", ", ")).toBe("a")
    expect(removeFromPrompt("a, <lora:wc:-0.5>, b", "<lora:wc:1>", ", ")).toBe("a, b")
    expect(removeFromPrompt("a <lora:other:1>", "<lora:wc:1>", " ")).toBeNull()
  })

  it("keeps activation text the user moved away from the tag", () => {
    expect(removeFromPrompt("trigger, a <lora:wc:1>", "<lora:wc:1> trigger", " ")).toBe("trigger, a")
  })

  it("toggles plain text as a whole word", () => {
    expect(togglePromptText("lowres, easynegative", "easynegative", ", ")).toEqual({ value: "lowres", added: false })
    expect(togglePromptText("easynegativeV2", "easynegative", " ")).toEqual({ value: "easynegativeV2 easynegative", added: true })
    expect(removeFromPrompt("a easynegative b easynegative", "easynegative", " ")).toBe("a b")
    expect(removeFromPrompt("(blurry:1.2), x", "(blurry:1.2)", ", ")).toBe("x")
  })

  it("tells whether a card is in the prompt", () => {
    expect(isInPrompt("a <lora:wc:0.3> b", "<lora:wc:1> trigger")).toBe(true)
    expect(isInPrompt("a <lora:wcx:0.3> b", "<lora:wc:1>")).toBe(false)
    expect(isInPrompt("bad_hands, x", "bad_hands")).toBe(true)
    expect(isInPrompt("very_bad_hands", "bad_hands")).toBe(false)
    expect(isInPrompt("", "x")).toBe(false)
    expect(isInPrompt("x", null)).toBe(false)
  })
})
