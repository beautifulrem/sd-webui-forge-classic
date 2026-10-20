/**
 * The img2img form as the stage editor sees it (real schema): the binding, the layer each mode
 * edits, the editor props derived from the values, and the source / size writes.
 */
import { QueryClientProvider } from "@tanstack/react-query"
import { act, renderHook } from "@testing-library/react"
import type { ReactNode } from "react"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import img2img from "@/api/mock/contract/img2img.json"
import { keys, queryClient } from "@/api/queries"
import type { Bootstrap } from "@/api/types"
import { loadSchema } from "@/pages/generate/use-form-loader"
import { bootstrapWith } from "@/test/api-fixtures"
import { getFormStore } from "@/ui/form-store"
import type { TabSchema } from "@/ui/schema"
import { editorBinding } from "../core/modes"
import { setSizeFromImage, setSourceImage, useImg2imgBinding, useImg2imgEditorProps, useModeLayers } from "./use-img2img"

const schema = img2img as unknown as TabSchema
const binding = editorBinding(schema.core.layout)!
const store = () => getFormStore("img2img")
const core = () => store().getState().values.core ?? {}
const set = (values: Record<string, unknown>) => act(() => store().getState().setValues("core", values))
const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>

/** The mode layers and the editor props, as the stage reads them. */
const renderEditor = () =>
  renderHook(
    () => {
      const layers = useModeLayers(binding)
      return { layers, props: useImg2imgEditorProps(binding, layers) }
    },
    { wrapper },
  )

beforeEach(() => {
  store().setState({ revision: 0 })
  loadSchema(store(), schema)
})
afterEach(() => queryClient.clear())

describe("img2img binding", () => {
  it("is null until the schema is cached, then binds the source and its layers", () => {
    queryClient.setQueryDefaults(keys.schema("img2img"), { enabled: false })
    const { result, rerender } = renderHook(() => useImg2imgBinding(), { wrapper })
    expect(result.current).toBeNull()
    act(() => queryClient.setQueryData(keys.schema("img2img"), schema))
    rerender()
    expect(result.current?.background).toBe("init_img")
    expect(result.current?.layers.map((layer) => [layer.param, layer.kind, layer.painted])).toEqual([
      ["sketch", "sketch", true],
      ["mask", "mask", true],
      ["mask_upload", "mask", false],
    ])
  })
})

describe("mode layers", () => {
  it("names the layer each mode edits and where the inpaint mask comes from", () => {
    const { result } = renderHook(() => useModeLayers(binding))
    const expected: Record<string, [boolean, string | null, string | null]> = {
      img2img: [true, null, null],
      Sketch: [true, "sketch", null],
      Inpaint: [true, "mask", "painted"],
      "Inpaint sketch": [true, "sketch", "sketch"],
      "Inpaint upload": [true, "mask_upload", "uploaded"],
      Batch: [false, null, null],
    }
    for (const [mode, [canvas, active, mask]] of Object.entries(expected)) {
      set({ mode })
      expect([result.current.canvas, result.current.active?.param ?? null, result.current.mask], mode).toEqual([canvas, active, mask])
    }
  })

  it("keeps the same object while other values change, and is empty without a binding", () => {
    set({ mode: "Inpaint" })
    const { result } = renderHook(() => useModeLayers(binding))
    const before = result.current
    set({ prompt: "a fox", width: 768 })
    expect(result.current).toBe(before)
    expect(renderHook(() => useModeLayers(null)).result.current).toEqual({ canvas: false, active: null, mask: null })
  })
})

describe("editor props", () => {
  it("carry the image, the labelled layers, the active layer and its error", () => {
    set({ mode: "Inpaint", init_img: "src", sketch: null, mask: "m1", mask_upload: "u1" })
    const { result } = renderEditor()
    expect(result.current.props).toMatchObject({
      image: "src",
      activeLayer: "mask",
      error: null,
      layers: {
        sketch: { kind: "sketch", value: null, label: "Sketch" },
        mask: { kind: "mask", value: "m1", label: "Mask" },
        mask_upload: { kind: "mask", value: "u1", label: "Uploaded mask" },
      },
    })
    act(() => store().getState().setErrors([{ owner: "core", param: "mask", message: "Draw a mask first" }]))
    expect(result.current.props.error).toBe("Draw a mask first")
  })

  it("frame the output size, or the image times the scale in Resize by", () => {
    set({ mode: "img2img", resize_tab: "Resize to", width: 832, height: 1216, resize_mode: "Crop and Resize" })
    const { result } = renderEditor()
    expect(result.current.props.frame).toEqual({ width: 832, height: 1216, resizeMode: "Crop and Resize" })
    expect(result.current.props.blankSize).toEqual({ width: 832, height: 1216 })
    expect(result.current.props.inpaint).toBeNull()

    set({ resize_tab: "Resize by", scale_by: 1.5 })
    expect(result.current.props.frame).toEqual({ width: 0, height: 0, resizeMode: "Crop and Resize", scale: { by: 1.5, step: 64 } })

    set({ resize_tab: "Resize to", width: 0 })
    expect(result.current.props.frame).toBeNull()
    expect(result.current.props.blankSize).toBeNull()
  })

  it("pass the inpaint settings while a mask is in use", () => {
    set({ mode: "Inpaint", mask_blur: 8, inpaint_full_res: "Only masked", inpaint_full_res_padding: 32, inpainting_mask_invert: "Inpaint not masked" })
    const { result } = renderEditor()
    expect(result.current.props.inpaint).toEqual({ blur: 8, padding: 32, onlyMasked: true, inverted: true })
    set({ inpaint_full_res: "Whole picture", inpainting_mask_invert: "Inpaint masked" })
    expect(result.current.props.inpaint).toEqual({ blur: 8, padding: 32, onlyMasked: false, inverted: false })
  })

  it("are empty without a binding", () => {
    const { result } = renderHook(() => useImg2imgEditorProps(null, { canvas: false, active: null, mask: null }))
    expect(result.current).toMatchObject({ image: null, layers: {}, activeLayer: null, inpaint: null })
  })
})

describe("source and size writes", () => {
  it("takes the new image's size only with img2img_autosize", async () => {
    set({ width: 512, height: 512, resize_tab: "Resize by" })
    queryClient.setQueryData<Bootstrap>(keys.app, bootstrapWith({ img2img_autosize: false }))
    await setSourceImage(binding, "a", { width: 1024, height: 768 })
    expect(core()).toMatchObject({ init_img: "a", width: 512, height: 512 })

    queryClient.setQueryData<Bootstrap>(keys.app, bootstrapWith({ img2img_autosize: true }))
    await setSourceImage(binding, "b", { width: 1024, height: 768 })
    expect(core()).toMatchObject({ init_img: "b", width: 1024, height: 768, resize_tab: "Resize to" })
    // removing the image, or an image of unknown size, keeps the size
    await setSourceImage(binding, "c", { width: 0, height: 0 })
    await setSourceImage(binding, null, { width: 640, height: 640 })
    expect(core()).toMatchObject({ init_img: null, width: 1024, height: 768 })
  })

  it("snaps the image size to the sliders' step and bounds, including action overrides", () => {
    setSizeFromImage({ width: 5000, height: 10 })
    expect(core()).toMatchObject({ width: 2048, height: 64 })
    // an action changed the width slider's step and maximum
    store().setState((s) => ({ overrides: { ...s.overrides, core: { ...s.overrides.core, width: { step: 8, maximum: 4096 } } } }))
    setSizeFromImage({ width: 3001, height: 700 })
    expect(core()).toMatchObject({ width: 3000, height: 704 })
  })
})
