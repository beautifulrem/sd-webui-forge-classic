import { describe, expect, it } from "vitest"
import fixture from "./weights.fixture.json"
import { formatWeight, gridOf, parseLayerWeights, parseModuleWeights, serializeLayerWeights, serializeLayout, serializeModuleWeights } from "./weights"

describe("parseLayerWeights", () => {
  it.each(fixture.layers)("reads $text as Python's parse_layer_weights", ({ text, fallback, values }) => {
    const parsed = parseLayerWeights(text)
    expect(parsed.fallback).toBe(fallback)
    expect(Object.fromEntries([...parsed.values].map(([index, value]) => [String(index), value]))).toEqual(values)
  })

  it("lists what a run ignores", () => {
    expect(parseLayerWeights("0.5, 3=0.2").ignored).toEqual(["0.5"])
    expect(parseLayerWeights("12=abc, 300-400=1, x").ignored).toEqual(["12=abc", "300-400=1", "x"])
    expect(parseLayerWeights("all=1.0, 0-9=0.6").ignored).toEqual([])
  })
})

describe("parseModuleWeights", () => {
  // The grid anima-lora-layer-weight shows: every module module_name_from_block_tail names
  const MODULES = ["self_attn", "cross_attn", "mlp", "adaln", "norm"]
  it.each(fixture.modules)("reads $text as Python's parse_module_weights", ({ text, values }) => {
    const parsed = parseModuleWeights(text, MODULES)
    const recorded: Record<string, number | undefined> = values
    for (const [module, value] of Object.entries(recorded)) {
      // A module the grid does not show is still applied by a run: kept, not ignored
      if (MODULES.includes(module)) expect(parsed.values[module]).toBe(value)
      else expect(parsed.kept.some((entry) => entry.toLowerCase().startsWith(module))).toBe(true)
    }
    expect(parsed.written.toSorted()).toEqual(Object.keys(recorded).filter((module) => MODULES.includes(module)).toSorted())
  })

  it("separates kept entries from those a run ignores", () => {
    const parsed = parseModuleWeights("mlp=x, norm=0.3, final_layer=0.5, 0.7", ["mlp", "norm"])
    expect(parsed.values).toEqual({ mlp: 1, norm: 0.3 })
    expect(parsed.kept).toEqual(["final_layer=0.5"])
    expect(parsed.ignored).toEqual(["mlp=x", "0.7"])
  })
})

describe("serialization", () => {
  it("writes runs that differ from the fallback", () => {
    const grid = gridOf(parseLayerWeights("0-9=0.6,10-18=1.0,19-27=0.75"), 28)
    expect(serializeLayerWeights(1, grid)).toBe("all=1.0, 0-9=0.6, 19-27=0.75")
    expect(serializeLayerWeights(0.8, [0.8, 0.5, 0.8])).toBe("all=0.8, 1=0.5")
  })

  it("keeps the default text and an empty Qwen field", () => {
    expect(serializeLayerWeights(1, gridOf(parseLayerWeights("all=1.0"), 28))).toBe("all=1.0")
    expect(serializeLayerWeights(1, Array(28).fill(1), true)).toBe("")
    expect(serializeLayerWeights(1, [1, 0.3], true)).toBe("all=1.0, 1=0.3")
  })

  it("round-trips through the parser", () => {
    for (const text of ["all=0.9, 0-13=0.3", "all=1.0, 0=0.0, 5-7=1.25, 27=2.0"]) {
      const parsed = parseLayerWeights(text)
      expect(serializeLayerWeights(parsed.fallback, gridOf(parsed, 28))).toBe(text)
    }
  })

  it("formats weights like Python floats", () => {
    expect([1, 0.5, 2, 0.123456, -0.25].map(formatWeight)).toEqual(["1.0", "0.5", "2.0", "0.1235", "-0.25"])
  })

  it("writes the modules the text had or that differ from 1.0, then the kept entries", () => {
    const modules = ["self_attn", "mlp", "norm"]
    expect(serializeModuleWeights({ values: { self_attn: 1, mlp: 0.7, norm: 1 }, written: ["self_attn"], kept: [] }, modules)).toBe("self_attn=1.0\nmlp=0.7")
    const parsed = parseModuleWeights("mlp=0.5\nnorm=0.5\nfinal_layer=2", modules)
    expect(serializeModuleWeights({ ...parsed, values: { ...parsed.values, mlp: 0.6 } }, modules)).toBe("mlp=0.6\nnorm=0.5\nfinal_layer=2")
  })

  it("keeps the blocks past the layout a grid shows", () => {
    const parsed = parseLayerWeights("all=1.0, 30-39=0.5")
    expect(serializeLayout(parsed, gridOf(parsed, 28).with(0, 0.2), 1)).toBe("all=1.0, 0=0.2, 30-39=0.5")
    // "All others" moves the unlisted blocks past the layout with it
    expect(serializeLayout(parsed, Array(28).fill(0.8), 0.8)).toBe("all=0.8, 30-39=0.5")
  })
})
