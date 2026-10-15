import { describe, expect, it } from "vitest"
import { editAttention } from "./attention"
import { keyOptionsFrom, minimalChange } from "./editing"
import { moveTags } from "./reorder"

/**
 * Runs an edit on text whose cursor is marked `|` or selection `«…»`; the
 * result marks the new selection with `[…]`.
 */
function edit(marked: string, direction: 1 | -1) {
  let text = marked
  let from: number
  let to: number
  if (text.includes("«")) {
    from = text.indexOf("«")
    text = text.replace("«", "")
    to = text.indexOf("»")
    text = text.replace("»", "")
  } else {
    from = to = text.indexOf("|")
    text = text.replace("|", "")
  }
  const result = editAttention(text, from, to, direction)
  if (!result) return null
  return `${result.text.slice(0, result.from)}[${result.text.slice(result.from, result.to)}]${result.text.slice(result.to)}`
}

describe("editAttention", () => {
  it("wraps the word under the cursor", () => {
    expect(edit("a cat, blue s|ky", 1)).toBe("a cat, blue ([sky]:1.1)")
    expect(edit("a cat, blue s|ky", -1)).toBe("a cat, blue ([sky]:0.9)")
  })

  it("wraps a selection, trimming trailing spaces", () => {
    expect(edit("«blue sky », sun", 1)).toBe("([blue sky]:1.1) , sun")
  })

  it("steps an existing weight and unwraps at 1.0", () => {
    expect(edit("(s|ky:1.1)", 1)).toBe("([sky]:1.2)")
    expect(edit("(s|ky:1.1)", -1)).toBe("[sky]")
  })

  it("selects the enclosing block", () => {
    expect(edit("(blue |sky:1.5), sun", -1)).toBe("([blue sky]:1.4), sun")
  })

  it("converts legacy emphasis", () => {
    expect(edit("((s|ky))", 1)).toBe("([sky]:1.3)")
    expect(edit("[s|ky]", -1)).toBe("([sky]:0.8)")
  })

  it("uses the extra-network precision inside <…>", () => {
    expect(edit("<lora:st|yle:0.8>", 1)).toBe("<[lora:style]:0.85>")
    expect(edit("<lora:st|yle:1.0>", -1)).toBe("<[lora:style]:0.95>")
  })

  it("keeps integers formatted as x.0", () => {
    expect(edit("(s|ky:1.9)", 1)).toBe("([sky]:2.0)")
  })

  it("does nothing on empty text", () => {
    expect(edit("|", 1)).toBeNull()
  })
})

describe("moveTags", () => {
  const move = (text: string, cursor: number, direction: -1 | 1) => moveTags(text, cursor, cursor, direction)

  it("moves the tag under the cursor", () => {
    expect(move("a, b, c", 4, -1)?.text).toBe(" b,a, c")
    expect(move("a, b, c", 4, 1)?.text).toBe("a, c, b")
  })

  it("does not move past the ends", () => {
    expect(move("a, b, c", 0, -1)).toBeNull()
    expect(move("a, b, c", 7, 1)).toBeNull()
  })

  it("moves a multi-tag selection and keeps it selected", () => {
    const result = moveTags("a, b, c, d", 3, 6, 1)!
    expect(result.text).toBe("a, d, b, c")
    expect(result.text.slice(result.from, result.to)).toBe(" b, c")
  })
})

describe("prompt editor helpers", () => {
  it("computes a minimal single replacement", () => {
    expect(minimalChange("a (cat:1.1) b", "a (cat:1.2) b")).toEqual({ from: 9, to: 10, insert: "2" })
    expect(minimalChange("abc", "abc")).toEqual({ from: 3, to: 3, insert: "" })
    expect(minimalChange("", "xyz")).toEqual({ from: 0, to: 0, insert: "xyz" })
  })

  it("reads keyedit options with defaults", () => {
    expect(keyOptionsFrom(undefined).attention.precisionAttention).toBe(0.1)
    const options = keyOptionsFrom({ keyedit_precision_attention: 0.05, keyedit_move: false })
    expect(options.attention.precisionAttention).toBe(0.05)
    expect(options.move).toBe(false)
  })
})
