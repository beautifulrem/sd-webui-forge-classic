import { describe, expect, it } from "vitest"
import { compilePrompt, DraftError, escapeParens, parseUniqueJson } from "./compile"
import { COMPILED, DRAFT } from "./testdata"

const errorOf = (input: unknown) => {
  try {
    compilePrompt(input)
  } catch (error) {
    expect(error).toBeInstanceOf(DraftError)
    return (error as Error).message
  }
  throw new Error("compiled")
}

const draftWith = (patch: Record<string, unknown>) => ({ ...structuredClone(DRAFT), ...patch })
const itemWith = (patch: Record<string, unknown>) => draftWith({ items: [{ ...DRAFT.items[0], ...patch }] })

describe("compilePrompt", () => {
  it("compiles a Layout draft exactly as upstream (centred yxyx boxes, outward rounding, escaped parentheses)", () => {
    expect(compilePrompt(DRAFT)).toBe(COMPILED)
    expect(compilePrompt(JSON.stringify(DRAFT))).toBe(COMPILED)
  })

  it("compiles a caption draft to its escaped caption", () => {
    expect(compilePrompt('{"caption": "a (red) hat"}')).toBe("a \\(red\\) hat")
  })

  it("folds backslashes before a parenthesis into one escape", () => {
    expect(escapeParens("a \\\\(b) \\(c\\)")).toBe("a \\(b\\) \\(c\\)")
  })

  it("keeps optional item and summary fields (tags, palette) in the prompt", () => {
    const prompt = compilePrompt(draftWith({ summary: { caption: "x", palette: ["red"] }, items: [{ ...DRAFT.items[0], tags: ["cute"] }] }))
    expect(JSON.parse(prompt)).toMatchObject({ summary: { caption: "x", palette: ["red"] }, items: [{ tags: ["cute"] }] })
  })

  it.each([
    [draftWith({ width: 1000 }), "width must be a positive integer multiple of 16"],
    [draftWith({ height: 0 }), "height must be a positive integer multiple of 16"],
    [draftWith({ bbox_coord_space: "rel" }), "Draft bbox_coord_space must be abs"],
    [draftWith({ bbox_format: "yxyx" }), "Draft bbox_format must be xyxy"],
    [draftWith({ extra: 1 }), 'Layout draft: unknown fields ["extra"]'],
    [{ width: 16 }, 'Layout draft: missing fields ["bbox_coord_space","bbox_format","height","items","summary"]'],
    [draftWith({ summary: { caption: " " } }), "summary.caption must be nonempty text"],
    [draftWith({ summary: { caption: "x", canvas: "grid" } }), "summary.canvas must be single, multi_panel or collage"],
    [draftWith({ summary: { caption: "x", meta: ["a", ""] } }), "summary.meta[1] must be nonempty text"],
    [draftWith({ items: [] }), "items must be a nonempty flat array"],
    [itemWith({ label: "thing" }), "items[0].label must be object, character or text"],
    [itemWith({ caption: "" }), "items[0].caption must be nonempty text"],
    [itemWith({ label: "text" }), "items[0]: text items require literal text"],
    [itemWith({ text: " " }), "items[0].text must be nonempty text"],
    [itemWith({ bbox: [1, 2, 3] }), "items[0].bbox must be [x1,y1,x2,y2]"],
    [itemWith({ bbox: [500, 200, 100, 700] }), "items[0].bbox must have positive width and height"],
    [itemWith({ bbox: [100, 200, 1100, 700] }), "items[0].bbox exceeds the 1024x768 canvas"],
    [itemWith({ bbox: [100, 200, 100.0000000001, 700] }), "items[0].bbox is too small at export precision"],
    [{ caption: "x", width: 16 }, 'Caption draft: unknown fields ["width"]'],
    [[], "Draft must be a JSON object"],
  ])("refuses %j: %s", (draft, message) => {
    expect(errorOf(draft)).toBe(message)
  })

  it("refuses duplicate keys, overflowing numbers and malformed JSON", () => {
    expect(errorOf('{"caption":"x","caption":"y"}')).toBe("Duplicate JSON key: caption")
    expect(errorOf('{"caption":"x","n":1e999}')).toBe("Invalid JSON number: 1e999")
    expect(errorOf('{"caption":"x"} x')).toBe("Invalid JSON")
    expect(errorOf("[1,2")).toBe("Invalid JSON")
  })

  it("parses JSON literals (the field checks refuse them, not the parser)", () => {
    expect(parseUniqueJson('{"a":[true,false,null,-1.5e3]}')).toEqual({ a: [true, false, null, -1500] })
    expect(errorOf('{"caption":true}')).toBe("caption must be nonempty text")
  })
})
