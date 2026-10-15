import { describe, expect, it } from "vitest"
import { evaluate, jsonEqual, pyEqual, references, splitRef, truthy } from "./conditions"
import type { Condition } from "./schema"

const values: Record<string, Record<string, unknown>> = {
  core: { mode: "Inpaint", cfg_scale: 7, steps: 0, prompt: "", tags: [], enable_hr: true, script: "X/Y/Z plot" },
  anima: { enable: true, preset: "detail", weights: [1, 2], opts: { a: 1 } },
  controlnet: { enabled: false },
}
const resolve = (owner: string, name: string) => values[owner]?.[name]
const run = (cond: Condition | null | undefined, owner = "core") => evaluate(cond, owner, resolve)

describe("splitRef", () => {
  it("resolves bare names to the current owner", () => {
    expect(splitRef("mode", "anima")).toEqual(["anima", "mode"])
  })
  it("splits on the first dot", () => {
    expect(splitRef("core.width", "anima")).toEqual(["core", "width"])
    expect(splitRef("legacy.#0", "x")).toEqual(["legacy", "#0"])
  })
})

describe("truthy", () => {
  it.each([
    [false, false],
    [null, false],
    [undefined, false],
    [0, false],
    ["", false],
    [[], false],
    [true, true],
    [1, true],
    ["a", true],
    [[0], true],
    [{}, true],
  ])("%j → %s", (value, expected) => expect(truthy(value)).toBe(expected))
})

describe("evaluate", () => {
  it("treats absent conditions and constants", () => {
    expect(run(undefined)).toBe(true)
    expect(run(null)).toBe(true)
    expect(run(true)).toBe(true)
    expect(run(false)).toBe(false)
  })

  it("evaluates truthiness leaves", () => {
    expect(run({ param: "enable_hr" })).toBe(true)
    expect(run({ param: "steps" })).toBe(false)
    expect(run({ param: "prompt" })).toBe(false)
    expect(run({ param: "tags" })).toBe(false)
  })

  it("evaluates eq / ne / in", () => {
    expect(run({ param: "mode", eq: "Inpaint" })).toBe(true)
    expect(run({ param: "mode", eq: "img2img" })).toBe(false)
    expect(run({ param: "mode", ne: "img2img" })).toBe(true)
    expect(run({ param: "mode", in: ["Inpaint", "Inpaint sketch"] })).toBe(true)
    expect(run({ param: "mode", in: ["Sketch"] })).toBe(false)
  })

  it("compares JSON structurally and strictly", () => {
    expect(run({ param: "weights", eq: [1, 2] }, "anima")).toBe(true)
    expect(run({ param: "opts", eq: { a: 1 } }, "anima")).toBe(true)
    expect(run({ param: "cfg_scale", eq: "7" })).toBe(false)
  })

  it("evaluates numeric comparisons", () => {
    expect(run({ param: "cfg_scale", gt: 1 })).toBe(true)
    expect(run({ param: "cfg_scale", gte: 7 })).toBe(true)
    expect(run({ param: "cfg_scale", lt: 7 })).toBe(false)
    expect(run({ param: "cfg_scale", lte: 7 })).toBe(true)
    // Non-numbers never compare true.
    expect(run({ param: "mode", gt: 0 })).toBe(false)
  })

  it("combines with all / any / not", () => {
    expect(run({ all: [{ param: "enable_hr" }, { param: "mode", eq: "Inpaint" }] })).toBe(true)
    expect(run({ all: [{ param: "enable_hr" }, { param: "steps" }] })).toBe(false)
    expect(run({ any: [{ param: "steps" }, { param: "enable_hr" }] })).toBe(true)
    expect(run({ not: { param: "steps" } })).toBe(true)
    expect(run({ all: [] })).toBe(true)
    expect(run({ any: [] })).toBe(false)
  })

  it("addresses other owners and core", () => {
    expect(run({ param: "core.script", eq: "X/Y/Z plot" }, "xyz")).toBe(true)
    expect(run({ param: "anima.enable" }, "controlnet")).toBe(true)
    expect(run({ param: "enabled" }, "controlnet")).toBe(false)
  })

  it("treats missing params as undefined", () => {
    expect(run({ param: "nope" })).toBe(false)
    expect(run({ param: "nope", eq: null })).toBe(false)
    expect(run({ param: "nope", in: [null] })).toBe(false)
    expect(run({ param: "nope", ne: "x" })).toBe(true)
    expect(run({ param: "ghost.param", ne: 1 })).toBe(true)
  })

  it("never hides controls because of a malformed condition", () => {
    expect(run({ bogus: 1 } as unknown as Condition)).toBe(true)
    expect(run({ all: "x" } as unknown as Condition)).toBe(true)
  })
})

describe("references", () => {
  it("lists every param a condition reads", () => {
    expect(references({ all: [{ param: "a" }, { not: { param: "core.b" } }, { any: [{ param: "s.c", eq: 1 }] }] }, "me")).toEqual([
      ["me", "a"],
      ["core", "b"],
      ["s", "c"],
    ])
  })
})

describe("jsonEqual", () => {
  it("distinguishes arrays and objects", () => {
    expect(jsonEqual([], {})).toBe(false)
    expect(jsonEqual({ a: [1, { b: 2 }] }, { a: [1, { b: 2 }] })).toBe(true)
    expect(jsonEqual({ a: 1 }, { a: 1, b: undefined })).toBe(false)
  })
})

describe("Python semantics (remi/ui/conditions.py)", () => {
  const values: Record<string, unknown> = { flag: true, off: false, n: 1, name: "beta", wide: "\uFF5E", none: null }
  const check = (cond: Parameters<typeof evaluate>[0]) => evaluate(cond, "me", (_owner, name) => values[name])

  it("compares booleans and numbers like Python (True == 1)", () => {
    expect(pyEqual(true, 1)).toBe(true)
    expect(pyEqual(false, 0)).toBe(true)
    expect(pyEqual(1, 1.0)).toBe(true)
    expect(pyEqual("1", 1)).toBe(false)
    expect(pyEqual([1, true], [true, 1])).toBe(true)
    expect(check({ param: "flag", eq: 1 })).toBe(true)
    expect(check({ param: "off", in: [0] })).toBe(true)
    expect(check({ param: "n", ne: true })).toBe(false)
  })

  it("orders strings by code point and numbers with booleans; mixed kinds are false", () => {
    expect(check({ param: "name", gt: "alpha" })).toBe(true)
    expect(check({ param: "name", lte: "alpha" })).toBe(false)
    expect(check({ param: "flag", gte: 1 })).toBe(true)
    expect(check({ param: "name", gt: 1 })).toBe(false)
    expect(check({ param: "none", lt: 5 })).toBe(false)
    expect(check({ param: "missing", lt: 5 })).toBe(false)
    // U+FF5E sorts before U+1F600 by code point (Python), after it by UTF-16 unit.
    expect(check({ param: "wide", lt: "\u{1F600}" })).toBe(true)
  })
})
