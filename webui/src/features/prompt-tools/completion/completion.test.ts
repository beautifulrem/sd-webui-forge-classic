import { describe, expect, it } from "vitest"
import type { CompletionItem } from "../api"
import { completionTarget, tokenEnd } from "./context"
import { commonPrefix, type InsertSettings, insertion, nextWildcardLevel, separatorAfter, tagText } from "./insert"

const at = (text: string) => {
  const cursor = text.indexOf("|")
  return { doc: text.replace("|", ""), cursor }
}

const target = (text: string) => {
  const { doc, cursor } = at(text)
  return completionTarget(doc, cursor)
}

describe("completion target", () => {
  it("finds the tag being typed", () => {
    expect(target("1girl, long ha|")).toEqual({ mode: "tag", query: "long ha", from: 7, to: 14 })
    expect(target("(long ha|")).toMatchObject({ mode: "tag", query: "long ha", from: 1 })
    expect(target("[cat:do|")).toMatchObject({ mode: "tag", query: "do" })
    expect(target("name_\\(se|")).toMatchObject({ mode: "tag", query: "name_\\(se", from: 0 })
    expect(target("a\nBREAK sm|")).toMatchObject({ mode: "tag", query: "sm", from: 8 })
    expect(target("*hai|")).toMatchObject({ query: "*hai" })
  })

  it("ignores weights, keywords, comments and empty words", () => {
    expect(target("(long hair:1.|")).toBeNull()
    expect(target("a, BREAK|")).toBeNull()
    expect(target("a, |")).toBeNull()
    expect(target("# notes ab|")).toBeNull()
    expect(target("/* off ab|")).toBeNull()
    expect(target("<lora:x:0.|")).toBeNull()
    expect(target("<lora:x:0.5> |")).toBeNull()
  })

  it("routes extra networks by prefix", () => {
    expect(target("a, <lora:sty|")).toEqual({ mode: "lora", query: "sty", from: 3, to: 12 })
    expect(target("<l:x|")).toMatchObject({ mode: "lora", query: "x" })
    expect(target("<e:bad|")).toMatchObject({ mode: "embedding", query: "bad" })
    expect(target("<chant:ba|")).toMatchObject({ mode: "chant", query: "ba" })
    expect(target("<lo|")).toMatchObject({ mode: "network", query: "" })
    expect(target("<sty|")).toMatchObject({ mode: "network", query: "sty" })
    expect(target("<hypernet:x|")).toBeNull()
  })

  it("finds wildcards and their values", () => {
    expect(target("a, __hair/co|")).toEqual({ mode: "wildcard", query: "hair/co", from: 3, to: 12 })
    expect(target("__|")).toMatchObject({ mode: "wildcard", query: "", from: 0 })
    expect(target("a __animals__ca|")).toEqual({ mode: "wildcard-value", query: "ca", from: 2, to: 15, wildcard: "animals" })
  })

  it("extends the replaced range over the rest of the token", () => {
    const { doc, cursor } = at("long ha|ir, b")
    expect(tokenEnd(doc, completionTarget(doc, cursor)!)).toBe(9)
    const lora = at("<lora:st|yle:1>, x")
    expect(tokenEnd(lora.doc, completionTarget(lora.doc, lora.cursor)!)).toBe(14)
  })
})

const SETTINGS: InsertSettings = {
  replace_underscores: true,
  underscore_exclusions: ["^_^"],
  escape_parentheses: true,
  append_comma: true,
  append_space: true,
  always_space_at_end: true,
  keywords: false,
  keyword_location: "Start of prompt",
  artist_prefix: false,
  source: "danbooru",
  network_separator: " ",
  default_weight: 1,
  wildcard_mode: "To next folder level",
}

const item = (fields: Partial<CompletionItem> & Pick<CompletionItem, "text">): CompletionItem => ({ type: "tag", ...fields })

function insert(text: string, completion: CompletionItem, settings: Partial<InsertSettings> = {}, key: "Tab" | "Enter" | null = "Enter", common?: string) {
  const { doc, cursor } = at(text)
  const found = completionTarget(doc, cursor)!
  const result = insertion({ doc, target: found, to: tokenEnd(doc, found), item: completion, settings: { ...SETTINGS, ...settings }, key, common })
  let out = doc
  for (const change of [...result.changes].sort((a, b) => b.from - a.from)) out = out.slice(0, change.from) + change.insert + out.slice(change.to)
  return { text: out.slice(0, result.cursor) + "|" + out.slice(result.cursor), reopen: result.reopen }
}

describe("insertion rules", () => {
  it("formats tags: underscores, exclusions, escapes, artist prefix", () => {
    expect(tagText(item({ text: "long_hair" }), SETTINGS)).toBe("long hair")
    expect(tagText(item({ text: "^_^" }), SETTINGS)).toBe("^_^")
    expect(tagText(item({ text: "name_(series)" }), SETTINGS)).toBe("name \\(series\\)")
    expect(tagText(item({ text: "name_(series)" }), { ...SETTINGS, escape_parentheses: false, replace_underscores: false })).toBe("name_(series)")
    expect(tagText(item({ text: "wlop", category: 1 }), { ...SETTINGS, artist_prefix: true })).toBe("@wlop")
    expect(tagText(item({ text: "wlop", category: 1 }), { ...SETTINGS, artist_prefix: true, source: "e621" })).toBe("wlop")
    expect(tagText(item({ text: "@wlop", category: 1 }), { ...SETTINGS, artist_prefix: true })).toBe("@wlop")
  })

  it("chooses the separator after a tag", () => {
    expect(separatorAfter("", SETTINGS)).toBe(", ")
    expect(separatorAfter(", next", SETTINGS)).toBe("")
    expect(separatorAfter(":1.2)", SETTINGS)).toBe("")
    expect(separatorAfter(")", SETTINGS)).toBe("")
    expect(separatorAfter(" next", SETTINGS)).toBe(",")
    expect(separatorAfter("", { ...SETTINGS, append_comma: false, append_space: false })).toBe(" ")
    expect(separatorAfter("", { ...SETTINGS, append_comma: false, append_space: false, always_space_at_end: false })).toBe("")
  })

  it("replaces the typed word and places the cursor after the separator", () => {
    expect(insert("1girl, long ha|", item({ text: "long_hair" })).text).toBe("1girl, long hair, |")
    expect(insert("(long ha|:1.2)", item({ text: "long_hair" })).text).toBe("(long hair|:1.2)")
    expect(insert("long ha|ir, smile", item({ text: "long_hair" })).text).toBe("long hair|, smile")
    expect(insert("x, ~ch|", item({ text: "chant text", type: "chant", label: "Chant" })).text).toBe("x, chant text, |")
  })

  it("writes LoRAs with their weight and activation text", () => {
    const lora = item({ type: "lora", text: "styleAlias", label: "folder/style", weight: 0.7, keywords: "trigger word" })
    expect(insert("1girl, <lora:sty|", lora).text).toBe("1girl, <lora:styleAlias:0.7>|")
    expect(insert("1girl, <sty|, smile", lora).text).toBe("1girl, <lora:styleAlias:0.7>|, smile")
    expect(insert("1girl, <lora:sty|", lora, { keywords: true }).text).toBe("trigger word, 1girl, <lora:styleAlias:0.7>|")
    expect(insert("1girl, <lora:sty|", lora, { keywords: true, keyword_location: "End of prompt" }).text).toBe("1girl, <lora:styleAlias:0.7>|, trigger word")
    expect(insert("a, <lora:sty|, b", lora, { keywords: true, keyword_location: "End of prompt" }).text).toBe("a, <lora:styleAlias:0.7>|, b, trigger word")
    expect(insert("1girl, <lora:sty|", lora, { keywords: true, keyword_location: "Before LORA/LyCO" }).text).toBe("1girl, trigger word, <lora:styleAlias:0.7>|")
    expect(insert("1girl, <lora:sty|", lora, { keywords: true, keyword_location: "After LORA/LyCO" }).text).toBe("1girl, <lora:styleAlias:0.7> trigger word|")
    // already there: not added twice
    expect(insert("trigger word, <lora:sty|", lora, { keywords: true }).text).toBe("trigger word, <lora:styleAlias:0.7>|")
  })

  it("completes nested wildcards a folder at a time with Tab", () => {
    expect(nextWildcardLevel("hair/colors/light", "ha", "To next folder level")).toBe("hair/")
    expect(nextWildcardLevel("hair/colors/light", "hair/", "To next folder level")).toBe("hair/colors/")
    expect(nextWildcardLevel("hair/colors/light", "hair/colors/l", "To next folder level")).toBe("hair/colors/light")
    expect(nextWildcardLevel("hair/colors/light", "h", "To first difference", "hair/col")).toBe("hair/col")
    expect(nextWildcardLevel("hair/colors/light", "h", "Always fully")).toBe("hair/colors/light")
    expect(commonPrefix(["hair/colors/light", "hair/colors/dark", "hair/cut"])).toBe("hair/c")
    const wildcard = item({ type: "wildcard", text: "hair/colors/light" })
    expect(insert("a, __ha|", wildcard, {}, "Tab")).toEqual({ text: "a, __hair/|", reopen: true })
    expect(insert("a, __ha|", wildcard, {}, "Enter")).toEqual({ text: "a, __hair/colors/light__|", reopen: true })
    expect(insert("a __animals__ca|", item({ type: "wildcard-value", text: "cat" })).text).toBe("a cat, |")
  })
})

describe("Anima's @ artist prefix", () => {
  it("is not searched but stays in the replaced range", () => {
    expect(target("1girl, @wl|")).toEqual({ mode: "tag", query: "wl", from: 7, to: 10, prefix: "@" })
    expect(target("(@as|")).toMatchObject({ query: "as", from: 1, prefix: "@" })
    // an emoticon tag, not a prefix
    expect(target("@_|")).toEqual({ mode: "tag", query: "@_", from: 0, to: 2 })
  })

  it("is kept for an artist, dropped for other tags", () => {
    expect(insert("1girl, @wl|", item({ text: "wlop", category: 1 })).text).toBe("1girl, @wlop, |")
    expect(insert("1girl, @wl|", item({ text: "wlop", category: 1 }), { artist_prefix: true }).text).toBe("1girl, @wlop, |")
    expect(insert("1girl, @lo|", item({ text: "long_hair", category: 0 })).text).toBe("1girl, long hair, |")
    expect(insert("@wl|", item({ text: "wlop", category: 1 }), { source: "derpibooru" }).text).toBe("wlop, |")
  })
})

describe("prompt comments as Forge strips them", () => {
  it("treats every # and // as a line comment", () => {
    expect(target("see http://example.com/a, ta|")).toBeNull()
    expect(target("a \\# not escaped ta|")).toBeNull()
    expect(target("/* closed */ ta|")).toMatchObject({ mode: "tag", query: "ta" })
    expect(target("# note\nlong ha|")).toMatchObject({ mode: "tag", query: "long ha" })
  })
})
