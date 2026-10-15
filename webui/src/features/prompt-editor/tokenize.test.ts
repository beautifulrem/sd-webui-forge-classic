import { describe, expect, it } from "vitest"
import { bracketProblems, type PromptTokenKind, tokenizePrompt } from "./tokenize"

/** `[kind, text]` of every token, for readable expectations. */
function kinds(text: string, only?: PromptTokenKind[]) {
  return tokenizePrompt(text)
    .filter((token) => !only || only.includes(token.kind))
    .map((token) => [token.kind, text.slice(token.from, token.to)])
}

describe("tokenizePrompt", () => {
  it("marks attention weights and bracket depth", () => {
    const tokens = tokenizePrompt("((a:1.2), b)")
    expect(tokens.filter((t) => t.kind === "bracket").map((t) => t.depth)).toEqual([0, 1, 1, 0])
    expect(kinds("(cat: 1.25 ) and (dog)", ["weight"])).toEqual([["weight", ": 1.25 "]])
  })

  it("reads schedules and alternations", () => {
    expect(kinds("[cat:dog:0.4]", ["schedule", "weight"])).toEqual([
      ["schedule", ":"],
      ["schedule", ":"],
      ["weight", "0.4"],
    ])
    expect(kinds("[red|blue] {a|b}", ["schedule"])).toEqual([
      ["schedule", "|"],
      ["schedule", "|"],
    ])
  })

  it("treats extra networks as one token with numeric arguments", () => {
    expect(kinds("x <lora:style (v2):0.8:0.5> y", ["extra", "extra-weight", "bracket", "bracket-error"])).toEqual([
      ["extra", "<lora:style (v2):0.8:0.5>"],
      ["extra-weight", "0.8"],
      ["extra-weight", "0.5"],
    ])
    // Emoticons are not extra networks.
    expect(kinds(">_< smile", ["extra"])).toEqual([])
  })

  it("finds keywords, wildcards, escapes and comment lines", () => {
    expect(kinds("a BREAK b AND c BREAKS ANDROID", ["keyword"])).toEqual([
      ["keyword", "BREAK"],
      ["keyword", "AND"],
    ])
    expect(kinds("__colors__ and my__thing__", ["wildcard"])).toEqual([["wildcard", "__colors__"]])
    expect(kinds("\\(literal\\)", ["escape", "bracket", "bracket-error"])).toEqual([
      ["escape", "\\("],
      ["escape", "\\)"],
    ])
    expect(kinds("# a note (\ncat", ["comment", "bracket-error"])).toEqual([["comment", "# a note ("]])
  })
})

describe("bracketProblems", () => {
  it("is empty for balanced prompts", () => {
    expect(bracketProblems("((masterpiece:1.2)), [a:b:0.5], {x|y}, <lora:n:1>")).toEqual([])
  })

  it("reports unclosed and unmatched brackets with positions", () => {
    expect(bracketProblems("(cat, dog")).toEqual([{ from: 0, to: 1, char: "(", kind: "unclosed" }])
    expect(bracketProblems("cat)")).toEqual([{ from: 3, to: 4, char: ")", kind: "unmatched" }])
  })

  it("reports a mismatched pair on both sides", () => {
    const problems = bracketProblems("(cat]")
    expect(problems.map((p) => [p.char, p.kind])).toEqual([
      ["(", "unclosed"],
      ["]", "unmatched"],
    ])
  })
})
