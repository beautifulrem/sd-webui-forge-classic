import { describe, expect, it } from "vitest"
import fixture from "./model.fixture.json"
import { type Curve, factorAt, resolve, type Stage, type Template, timingOf } from "./model"

// The props anima-lora-stage-scheduler gives the widget
const CURVES: Curve[] = [
  { value: "平滑淡入淡出", label: "Smooth fade in/out", shape: "smooth" },
  { value: "保持", label: "Hold", shape: "hold" },
  { value: "线性三角", label: "Linear triangle", shape: "triangle" },
  { value: "前强后弱", label: "Strong early, fading", shape: "front" },
  { value: "后强前弱", label: "Rising, strong late", shape: "back" },
]
const STAGES: Stage[] = [
  { value: "构图阶段", label: "Composition stage", kind: "composition" },
  { value: "人物特征阶段", label: "Character stage", kind: "character" },
  { value: "画风阶段", label: "Style stage", kind: "style" },
]
const templates = fixture.templates as Record<string, Template>

describe("the scheduled strength", () => {
  it.each(fixture.cases)("matches _stage_curve_factor for $template (auto $timing.auto, shift $shift)", ({ template, timing, shift, factors }) => {
    const resolved = resolve(timingOf(timing, "平滑淡入淡出"), templates[template], shift, STAGES, CURVES)
    fixture.points.forEach((x, i) => expect(factorAt(resolved, x, CURVES)).toBeCloseTo(factors[i]!, 12))
  })

  it("reads a partial value with the defaults", () => {
    expect(timingOf({ start: 0.3, curve: "" }, "保持")).toEqual({ auto: true, start: 0.3, end: 1, peak: 0.86, strength: 1, curve: "保持" })
    expect(timingOf(null, "保持").auto).toBe(true)
  })
})
