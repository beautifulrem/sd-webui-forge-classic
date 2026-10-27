import { describe, expect, it } from "vitest"
import { compilePrompt } from "./compile"
import { COMPILED, DRAFT } from "./testdata"
import { decompilePrompt, restoreJsonParens, splitPrompt } from "./decompile"

const errorOf = (input: unknown, width = 1024, height = 768) => {
  try {
    decompilePrompt(input, width, height)
  } catch (error) {
    return (error as Error).message
  }
  throw new Error("decompiled")
}

describe("decompilePrompt", () => {
  it("rebuilds the draft of a compiled prompt for its size (as upstream: boxes outward-rounded, escapes kept)", () => {
    expect(decompilePrompt(COMPILED, 1024, 768)).toEqual({
      width: 1024,
      height: 768,
      summary: { caption: "A cat \\(orange\\) on a sofa", meta: ["photo", "soft light"], canvas: "single" },
      items: [
        { label: "character", caption: "an orange cat", bbox: [100, 200, 501, 700] },
        { label: "text", caption: "a neon sign", bbox: [600, 50, 1000, 150], text: "OPEN \\(24h\\)" },
      ],
    })
    // Another size moves the boxes about the other centre.
    expect(decompilePrompt(COMPILED, 1000, 700).items[0]!.bbox).toEqual([88, 166, 489, 666])
  })

  it("round-trips: the rebuilt draft compiles to the same prompt (the builder's own order is checked in draft.test.ts)", () => {
    const draft = decompilePrompt(COMPILED, 1024, 768)
    const parse = (prompt: string) => JSON.parse(restoreJsonParens(prompt)) as unknown
    expect(parse(compilePrompt({ ...draft, bbox_coord_space: "abs", bbox_format: "xyxy" }))).toEqual(parse(COMPILED))
  })

  it("takes a saved draft as it is, whatever size is given, and fills the defaults", () => {
    const draft = decompilePrompt(JSON.stringify({ ...DRAFT, summary: { caption: "x" } }), Number.NaN, Number.NaN)
    expect(draft.width).toBe(1024)
    expect(draft.summary).toEqual({ caption: "x", meta: [], canvas: "single" })
    // Fields the builder does not edit are dropped (text only on text items).
    expect(decompilePrompt({ ...DRAFT, items: [{ ...DRAFT.items[0], text: "t", tags: ["a"] }] }, 0, 0).items[0]).toEqual({
      label: "character",
      caption: "an orange cat",
      bbox: [100.4, 200, 500.2, 700],
    })
  })

  it.each([
    ["[]", "Prompt must be a JSON object"],
    [COMPILED.replace('"yxyx"', '"xyxy"'), "Prompt bbox_format must be yxyx"],
    [COMPILED.replace('"abs_centralized"', '"abs"'), "Prompt bbox_coord_space must be abs_centralized"],
    [COMPILED.replace("[-184,-412,316,-11]", "[1,2]"), "items[0].bbox must be [y1,x1,y2,x2]"],
    [COMPILED.replace('"label":"character"', '"label":"thing"'), "items[0].label must be object, character or text"],
    [COMPILED.replace('"caption":"a neon sign"', '"caption":""'), "items[1].caption must be nonempty text"],
    [COMPILED.replace(',"text":"OPEN \\(24h\\)"', ""), "items[1].text must be nonempty text"],
    [COMPILED.replace('"meta":["photo","soft light"]', '"meta":[""]'), "summary.meta[0] must be nonempty text"],
    [COMPILED.replace('"canvas":"single"', '"canvas":"grid"'), "summary.canvas must be single, multi_panel or collage"],
  ])("refuses %s: %s", (input, message) => {
    expect(errorOf(input)).toBe(message)
  })

  it("needs a size for a compiled prompt", () => {
    expect(errorOf(COMPILED, Number.NaN, 768)).toBe("A valid width is required to decompile a prompt")
    expect(errorOf(COMPILED, 1024, 0)).toBe("A valid height is required to decompile a prompt")
  })

  it("throws a SyntaxError for text that is not JSON", () => {
    expect(() => decompilePrompt("a cat", 1024, 1024)).toThrow(SyntaxError)
  })
})

describe("restoreJsonParens", () => {
  it("turns prompt escapes into JSON escapes, leaving escaped backslashes alone", () => {
    expect(restoreJsonParens('"a \\(b\\)"')).toBe('"a \\\\(b\\\\)"')
    expect(JSON.parse(restoreJsonParens('"a \\(b\\)"'))).toBe("a \\(b\\)")
    expect(restoreJsonParens('"\\\\(x"')).toBe('"\\\\(x"')
  })
})

describe("splitPrompt", () => {
  it("separates the text added after the JSON", () => {
    expect(splitPrompt(`${COMPILED} <lora:moss:1>`)).toEqual({ json: COMPILED, extra: "<lora:moss:1>" })
    expect(splitPrompt(` ${COMPILED} `)).toEqual({ json: COMPILED, extra: "" })
  })

  it("is not fooled by braces and quotes inside strings", () => {
    const json = '{"summary":{"caption":"a } \\" { b"},"items":[]}'
    expect(splitPrompt(`${json} tail`)).toEqual({ json, extra: "tail" })
  })

  it("leaves other text whole", () => {
    expect(splitPrompt("a cat, {red|blue}")).toEqual({ json: "a cat, {red|blue}", extra: "" })
    expect(splitPrompt('{"a":')).toEqual({ json: '{"a":', extra: "" })
  })
})
