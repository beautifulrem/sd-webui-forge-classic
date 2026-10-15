import { describe, expect, it } from "vitest"
import { appendChips, canToggle, moveChip, parsePrompt, removeChip, serializePrompt, setChipText, setChipTexts, setChipWeight, toggleChip, toggleTag } from "./parse"

const ROUND_TRIP = [
  "",
  "   ",
  "masterpiece, best quality, 1girl",
  "  leading space, trailing comma,  ",
  "(masterpiece:1.2), ((long hair)), [blurry], (a, b:1.3)",
  "<lora:my_style:0.8>, <lora:other:0.6:0.4> 1girl <lyco:x:1>",
  "[cat:dog:0.5], {red|blue} eyes, a [b|c] d",
  "name_\\(series\\), \\(escaped\\) (nested (inner:1.1) outer:0.9)",
  "line one, a\nline two\n\nBREAK\nafter break AND other",
  "# a comment line\n1girl, /*disabled, */smile, // trailing comment\nend",
  "__hair/colors__, __animals__ cat",
  "unclosed (bracket, here",
  "emoji ^_^, >_<, http://example.com/a",
]

describe("prompt ↔ chips", () => {
  it.each(ROUND_TRIP)("round-trips %j exactly", (text) => {
    expect(serializePrompt(parsePrompt(text))).toBe(text)
  })

  it("reads weights, nesting and networks", () => {
    const chips = parsePrompt("(masterpiece:1.2), ((long hair)), [blurry], (a, b:1.3), <lora:style:0.8:0.5>, plain").chips
    expect(chips.map((c) => [c.kind, c.text, c.weight])).toEqual([
      ["tag", "masterpiece", 1.2],
      ["tag", "long hair", 1.21],
      ["tag", "blurry", 0.9091],
      ["tag", "a, b", 1.3],
      ["network", "style", 0.8],
      ["tag", "plain", 1],
    ])
    expect(chips[4]!.network).toBe("lora")
  })

  it("keeps schedules, alternations and escapes as plain tags", () => {
    const chips = parsePrompt("[cat:dog:0.5], [a|b], name_\\(series\\)").chips
    expect(chips.map((c) => [c.text, c.weight])).toEqual([
      ["[cat:dog:0.5]", 1],
      ["[a|b]", 1],
      ["name_\\(series\\)", 1],
    ])
  })

  it("splits BREAK / AND and comments into their own chips", () => {
    const chips = parsePrompt("a BREAK b, # note\nc /*d, */e").chips
    expect(chips.map((c) => [c.kind, c.text, c.disabled])).toEqual([
      ["tag", "a", false],
      ["keyword", "BREAK", false],
      ["tag", "b", false],
      ["comment", "note", true],
      ["tag", "c", false],
      ["tag", "d", true],
      ["tag", "e", false],
    ])
  })
})

describe("chip edits", () => {
  const doc = (text: string) => parsePrompt(text)

  it("sets weights canonically and keeps network arguments", () => {
    expect(setChipWeight(doc("a, ((b)), c"), 1, 1.3)).toBe("a, (b:1.3), c")
    expect(setChipWeight(doc("a, (b:1.3), c"), 1, 1)).toBe("a, b, c")
    expect(setChipWeight(doc("<lora:x:0.8:0.5>, y"), 0, 0.65)).toBe("<lora:x:0.65:0.5>, y")
    expect(setChipWeight(doc("<lora:x>"), 0, 0.7)).toBe("<lora:x:0.7>")
    expect(setChipWeight(doc("BREAK"), 0, 2)).toBe("BREAK")
  })

  it("edits text and keeps the weight", () => {
    expect(setChipText(doc("(cat:1.2), dog"), 0, "kitten")).toBe("(kitten:1.2), dog")
    expect(setChipText(doc("<lora:old:0.5>"), 0, "new")).toBe("<lora:new:0.5>")
    expect(setChipText(doc("a, b, c"), 1, "  ")).toBe("a, c")
  })

  it("disables a chip as a comment that takes its separator, and enables it back", () => {
    const text = "a, (b:1.2), c"
    const off = toggleChip(doc(text), 1)
    expect(off).toBe("a, /*(b:1.2), */c")
    expect(parsePrompt(off).chips[1]).toMatchObject({ disabled: true, text: "b", weight: 1.2 })
    expect(toggleChip(parsePrompt(off), 1)).toBe(text)
    const last = toggleChip(doc(text), 2)
    expect(last).toBe("a, (b:1.2)/*, c*/")
    expect(toggleChip(parsePrompt(last), 2)).toBe(text)
    expect(toggleChip(doc("solo"), 0)).toBe("/*solo*/")
    expect(toggleChip(parsePrompt("/*solo*/"), 0)).toBe("solo")
    // a disabled chip can be re-weighted
    expect(setChipWeight(parsePrompt(off), 1, 1.5)).toBe("a, /*(b:1.5), */c")
  })

  it("removes, moves and appends", () => {
    expect(removeChip(doc("a, b, c"), 1)).toBe("a, c")
    expect(removeChip(doc("a, b, c"), 2)).toBe("a, b")
    expect(removeChip(doc("a, b, c"), 0)).toBe("b, c")
    expect(removeChip(doc("a"), 0)).toBe("")
    expect(moveChip(doc("a, b\nc"), 2, 0)).toBe("c, a\nb")
    expect(moveChip(doc("a, <lora:x:0.5>, c"), 1, 2)).toBe("a, c, <lora:x:0.5>")
    expect(appendChips(doc("a, b"), "c")).toBe("a, b, c")
    expect(appendChips(doc("a, b, "), "c")).toBe("a, b, c")
    expect(appendChips(doc(""), "c")).toBe("c")
  })

  it("toggles a quick tag by its text", () => {
    expect(toggleTag("1girl, long hair", "long_hair")).toBe("1girl")
    expect(toggleTag("1girl", "smile")).toBe("1girl, smile")
  })
})

/** A small deterministic PRNG (mulberry32): the same prompts on every run. */
function random(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0
    let x = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296
  }
}

const PIECES = [
  "1girl", "long hair", "name_\\(series\\)", "(masterpiece:1.2)", "((nested (inner:1.1) outer))", "[faded]", "<lora:style:0.8>", "<lyco:x:1:0.5>",
  "[cat:dog:0.5]", "{a|b}", "BREAK", "AND", "__wild/card__", "/*off, */", "# note", "// note", "^_^", "@wlop", "a\\#b", "(unclosed", "closed)", "é ü 漢字",
]
const SEPARATORS = [", ", ",", " , ", "\n", ",\n", " ", "  ", ",, ", "\n\n"]

describe("prompt ↔ chips, randomised", () => {
  it("round-trips 500 random prompts exactly, and edits keep the rest intact", () => {
    const next = random(20261015)
    const pick = <T,>(list: readonly T[]) => list[Math.floor(next() * list.length)]!
    for (let run = 0; run < 500; run++) {
      let text = next() < 0.3 ? pick([" ", "\n", ", "]) : ""
      const count = 1 + Math.floor(next() * 8)
      for (let i = 0; i < count; i++) text += pick(PIECES) + (i < count - 1 || next() < 0.3 ? pick(SEPARATORS) : "")
      const doc = parsePrompt(text)
      expect(serializePrompt(doc), JSON.stringify(text)).toBe(text)
      // disabling an enabled chip keeps every other chip; enabling it again gives the prompt back
      const index = Math.floor(next() * doc.chips.length)
      const chip = doc.chips[index]
      if (chip && !chip.disabled && canToggle(chip)) {
        const disabled = parsePrompt(toggleChip(doc, index))
        expect(disabled.chips.map((c) => (c === disabled.chips[index] ? null : c.raw)), JSON.stringify(text)).toEqual(doc.chips.map((c) => (c === chip ? null : c.raw)))
        expect(disabled.chips[index]!.disabled).toBe(true)
        expect(toggleChip(disabled, index), JSON.stringify(text)).toBe(text)
      }
    }
  })
})

describe("comments as Forge strips them", () => {
  it("starts a line comment at any # or //, escaped or in a URL", () => {
    expect(parsePrompt("see http://x.com/a, b").chips.map((c) => [c.kind, c.raw])).toEqual([
      ["tag", "see http:"],
      ["comment", "//x.com/a, b"],
    ])
    expect(parsePrompt("a\\# b, c\nd").chips.map((c) => [c.kind, c.raw])).toEqual([
      ["tag", "a\\"],
      ["comment", "# b, c"],
      ["tag", "d"],
    ])
    // inside brackets too: the rest of the line is gone
    expect(parsePrompt("(a # b)\nc").chips.map((c) => c.kind)).toEqual(["tag", "comment", "tag"])
  })
})

describe("several chip edits at once", () => {
  it("replaces by the original indices, even when a new text has commas", () => {
    const doc = parsePrompt("金发, (猫:1.2), smile, /*狗, */end")
    expect(setChipTexts(doc, [
      { index: 0, text: "blonde hair, blue eyes" },
      { index: 1, text: "cat" },
      { index: 3, text: "dog" },
    ])).toBe("blonde hair, blue eyes, (cat:1.2), smile, /*dog, */end")
  })
})
