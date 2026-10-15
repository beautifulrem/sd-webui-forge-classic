import { describe, expect, it } from "vitest"
import contract from "./contract.json"
import { parsePose, poseText } from "./pose"
import {
  applyEdit,
  type Catalog,
  closestEight,
  defaultUnit,
  isPasted,
  mergePasted,
  modelsFor,
  modulesFor,
  normalizeUnit,
  PASTE_MARKER,
  selectModule,
  selectType,
  sliderValue,
} from "./unit"

const catalog = contract.props.catalog as Catalog

describe("unit values", () => {
  it("start where the server's do", () => {
    expect(PASTE_MARKER).toBe(contract.paste_marker)
    expect(defaultUnit(false)).toEqual(contract.default.txt2img)
    expect(defaultUnit(true)).toEqual(contract.default.img2img)
  })

  it("keep known keys of the right type", () => {
    const unit = normalizeUnit({ enabled: true, weight: "2", module: "canny", image: "up-1", sketch: 3, pose: { canvas_width: 1 }, extra: 1, [PASTE_MARKER]: true }, false)
    expect(unit).toEqual({ ...defaultUnit(false), enabled: true, module: "canny", image: "up-1", pose: { canvas_width: 1 } })
    expect(normalizeUnit(null, true)).toEqual(defaultUnit(true))
    expect(normalizeUnit([1], false)).toEqual(defaultUnit(false))
  })

  it("drop the preview when what it was made from changes", () => {
    const shown = { ...defaultUnit(false), image: "a", generated_image: "p", use_preview_as_input: true, pose: { x: 1 } }
    expect(applyEdit(shown, { weight: 0.5 })).toMatchObject({ generated_image: "p", use_preview_as_input: true })
    for (const patch of [{ module: "canny" }, { image: "b" }, { sketch: "s" }, { processor_res: 640 }, { threshold_a: 3 }, { pixel_perfect: true }]) {
      expect(applyEdit(shown, patch)).toMatchObject({ generated_image: null, use_preview_as_input: false, pose: null })
    }
    // Setting the same value, or a new preview with it, keeps it
    expect(applyEdit(shown, { image: "a" }).generated_image).toBe("p")
    expect(applyEdit(shown, { module: "canny", generated_image: "q" }).generated_image).toBe("q")
  })
})

describe("control type and preprocessor", () => {
  it("a control type selects its default preprocessor and model, with the sliders' values", () => {
    const unit = selectType(defaultUnit(false), "Canny", catalog)
    expect(unit).toMatchObject({ type_filter: "Canny", module: "canny", model: "control_v11p_sd15_canny", processor_res: 512, threshold_a: 100, threshold_b: 200 })
    expect(modulesFor(unit, catalog)).toEqual(["None", "canny"])
    expect(modelsFor(unit, catalog)).toEqual(["None", "control_v11p_sd15_canny", "t2i_union_promax"])
    const all = selectType(unit, "All", catalog)
    expect(all).toMatchObject({ type_filter: "All", module: "None", model: "None" })
  })

  it("lists the current value even when the type filters it out", () => {
    const unit = { ...defaultUnit(false), type_filter: "Canny", module: "depth_x", model: "depth_model" }
    expect(modulesFor(unit, catalog)).toContain("depth_x")
    expect(modelsFor(unit, catalog)).toContain("depth_model")
  })

  it("a preprocessor brings its slider defaults; unset sliders show them", () => {
    const unit = selectModule(defaultUnit(false), "reference_test", catalog)
    expect(unit.threshold_a).toBe(0.5)
    expect(sliderValue(-1, { visible: true, value: 7 })).toBe(7)
    expect(sliderValue(3, { visible: true, value: 7 })).toBe(3)
  })
})

describe("pasted units", () => {
  it("merge into the unit they replace, keeping its images and switch", () => {
    const current = { ...selectType(defaultUnit(false), "Canny", catalog), enabled: true, image: "img", image_fg: "mask", generated_image: "p" }
    const pasted = { [PASTE_MARKER]: true, module: "canny", weight: 0.4, guidance_end: 0.7 }
    expect(isPasted(pasted)).toBe(true)
    const merged = mergePasted(current, pasted, catalog)
    expect(merged).toMatchObject({ enabled: true, image: "img", image_fg: "mask", weight: 0.4, guidance_end: 0.7, type_filter: "Canny", generated_image: "p" })
    expect(PASTE_MARKER in merged).toBe(false)
  })

  it("show every preprocessor when the pasted one is not of the unit's type", () => {
    const current = selectType(defaultUnit(true), "Canny", catalog)
    const merged = mergePasted(current, { [PASTE_MARKER]: true, module: "reference_test" }, catalog)
    expect(merged).toMatchObject({ type_filter: "All", module: "reference_test", upload_independent: false })
  })
})

describe("helpers", () => {
  it("round sizes to the closest multiple of 8 (the old send-dimensions)", () => {
    expect([closestEight(512), closestEight(517), closestEight(516), closestEight(300), closestEight(301)]).toEqual([512, 520, 512, 296, 304])
  })

  it("read pose JSON", () => {
    expect(parsePose('{"canvas_width": 64, "canvas_height": 32, "people": []}')).toEqual({ ok: true, pose: { canvas_width: 64, canvas_height: 32, people: [] } })
    expect(parsePose('[{"canvas_width": 64, "canvas_height": 32}]').ok).toBe(true)
    expect(parsePose("{").ok).toBe(false)
    expect(parsePose('{"people": []}')).toEqual({ ok: false, error: "canvas_width is missing" })
    expect(poseText(null)).toBe("")
    expect(poseText({ a: 1 })).toBe('{\n  "a": 1\n}')
  })
})
