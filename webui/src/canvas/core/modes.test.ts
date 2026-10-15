import { describe, expect, it } from "vitest"
import img2img from "@/api/mock/contract/img2img.json"
import type { TabSchema } from "@/ui/schema"
import { editorBinding, modeLayers, modeParams } from "./modes"

const schema = img2img as unknown as TabSchema

describe("img2img modes → core params (real schema)", () => {
  const binding = editorBinding(schema.core.layout)!

  it("finds the background and the layers painted over it", () => {
    expect({ background: binding.background, layers: binding.layers }).toEqual({
      background: "init_img",
      layers: [
        { param: "sketch", kind: "sketch", painted: true },
        { param: "mask", kind: "mask", painted: true },
        { param: "mask_upload", kind: "mask", painted: false },
      ],
    })
  })

  const layersFor = (mode: string) => modeLayers(binding, (owner, name) => (owner === "core" && name === "mode" ? mode : schema.core.defaults[name]))

  it.each([
    ["img2img", { canvas: true, active: null, mask: null }, ["init_img"]],
    ["Sketch", { canvas: true, active: "sketch", mask: null }, ["init_img", "sketch"]],
    ["Inpaint", { canvas: true, active: "mask", mask: "painted" }, ["init_img", "mask"]],
    ["Inpaint sketch", { canvas: true, active: "sketch", mask: "sketch" }, ["init_img", "sketch"]],
    ["Inpaint upload", { canvas: true, active: "mask_upload", mask: "uploaded" }, ["init_img", "mask_upload"]],
    ["Batch", { canvas: false, active: null, mask: null }, []],
  ])("%s", (mode, expected, params) => {
    const layers = layersFor(mode)
    expect({ canvas: layers.canvas, active: layers.active?.param ?? null, mask: layers.mask }).toEqual(expected)
    expect(modeParams(layers, binding)).toEqual(params)
  })

  it("has no editor without an image input", () => {
    expect(editorBinding({ type: "column", children: [] })).toBeNull()
  })
})
