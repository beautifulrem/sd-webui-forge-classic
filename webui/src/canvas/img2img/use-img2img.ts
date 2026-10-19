/**
 * The img2img form as the editor sees it: which image params it binds
 * (from the schema), which layer the current mode edits, the values, and
 * the writes (user edits: errors cleared, change effects run).
 */
import { useQuery } from "@tanstack/react-query"
import { useMemo } from "react"
import { useStore } from "zustand"
import { clientOption } from "@/api/app-options"
import { schemaQuery } from "@/api/queries"
import { getFormStore } from "@/ui/form-store"
import type { SliderNode } from "@/ui/schema"
import { editorBinding, type EditorBinding, modeLayers, type ModeLayers } from "../core/modes"
import { replaceBackground, writeValues } from "../form/replace-background"
import type { EditorLayer, ImageEditorProps } from "../types"

export const IMG2IMG = "img2img"

const LAYER_LABELS: Record<string, string> = { sketch: "Sketch", mask: "Mask", mask_upload: "Uploaded mask" }

/** The img2img editor binding from the (cached) schema; null until it loads or without image inputs. */
export function useImg2imgBinding(): EditorBinding | null {
  const { data: schema } = useQuery(schemaQuery(IMG2IMG))
  const layout = schema?.core.layout ?? null
  return useMemo(() => (layout ? editorBinding(layout) : null), [layout])
}

const core = (s: ReturnType<ReturnType<typeof getFormStore>["getState"]>, name: string) => s.values.core?.[name]

/** Mode layers for the current values (re-renders only when they change). */
export function useModeLayers(binding: EditorBinding | null): ModeLayers {
  const store = getFormStore(IMG2IMG)
  const signature = useStore(store, (s) => {
    if (!binding) return "none"
    const layers = modeLayers(binding, (owner, name) => s.values[owner]?.[name])
    return `${layers.canvas ? 1 : 0}|${layers.active?.param ?? ""}|${layers.mask ?? ""}`
  })
  return useMemo(() => {
    const [canvas, active, mask] = signature.split("|")
    return {
      canvas: canvas === "1",
      active: binding?.layers.find((layer) => layer.param === active) ?? null,
      mask: (mask || null) as ModeLayers["mask"],
    }
  }, [signature, binding])
}

/** Writes core params as a user edit. */
export function setCoreValues(values: Record<string, unknown>) {
  writeValues({ store: getFormStore(IMG2IMG), formId: IMG2IMG, owner: "core" }, values)
}

/**
 * Sets the source image; the layers painted over the old one go with it
 * (undoable from a toast when they had paint, see `replaceBackground`).
 */
export async function setSourceImage(binding: EditorBinding, id: string | null, size?: { width: number; height: number } | null): Promise<void> {
  await replaceBackground(
    { store: getFormStore(IMG2IMG), formId: IMG2IMG, owner: "core", background: binding.background, painted: binding.layers.filter((layer) => layer.painted).map((layer) => layer.param) },
    id,
  )
  // "Automatically update the Width and Height when uploading image to img2img input"
  if (id && size && size.width > 0 && size.height > 0 && clientOption("img2img_autosize")) setSizeFromImage(size)
}

/** The size sliders' step (the server's resolution step), from the schema and its overrides. */
function sizeStep(state: ReturnType<ReturnType<typeof getFormStore>["getState"]>): number {
  const node = state.index.inputs.get("core")?.get("width") as SliderNode | undefined
  return state.overrides.core?.width?.step ?? node?.step ?? 8
}

/** Sets width and height from an image (snapped to the sliders' step, within bounds) and selects "Resize to". */
export function setSizeFromImage(size: { width: number; height: number }) {
  const store = getFormStore(IMG2IMG)
  const state = store.getState()
  const nodes = state.index.inputs.get("core")
  const fit = (name: "width" | "height", value: number) => {
    const node = nodes?.get(name) as SliderNode | undefined
    const override = state.overrides.core?.[name]
    const step = override?.step ?? node?.step ?? 8
    const min = override?.minimum ?? node?.minimum ?? 64
    const max = override?.maximum ?? node?.maximum ?? 4096
    return Math.min(max, Math.max(min, Math.round(value / step) * step))
  }
  const values: Record<string, unknown> = { width: fit("width", size.width), height: fit("height", size.height) }
  if (nodes?.has("resize_tab")) values.resize_tab = "Resize to"
  setCoreValues(values)
}

/** Everything the stage editor needs from the form, as `ImageEditor` props. */
export function useImg2imgEditorProps(binding: EditorBinding | null, layers: ModeLayers): Partial<ImageEditorProps> & { image: string | null } {
  const store = getFormStore(IMG2IMG)
  const params = binding ? [binding.background, ...binding.layers.map((layer) => layer.param)] : []
  const valuesKey = useStore(store, (s) => params.map((name) => String(core(s, name) ?? "")).join("|"))
  const errorsKey = useStore(store, (s) => {
    const names = [binding?.background, layers.active?.param].filter(Boolean) as string[]
    return names.map((name) => s.errors.core?.[name]).find(Boolean) ?? ""
  })
  const width = useStore(store, (s) => Number(core(s, "width") ?? 0))
  const height = useStore(store, (s) => Number(core(s, "height") ?? 0))
  const resizeMode = useStore(store, (s) => core(s, "resize_mode") as string | number | undefined)
  const resizeTab = useStore(store, (s) => core(s, "resize_tab"))
  const scaleBy = useStore(store, (s) => Number(core(s, "scale_by") ?? 1))
  const step = useStore(store, sizeStep)
  const blur = useStore(store, (s) => Number(core(s, "mask_blur") ?? 0))
  const padding = useStore(store, (s) => Number(core(s, "inpaint_full_res_padding") ?? 0))
  const onlyMasked = useStore(store, (s) => {
    const value = core(s, "inpaint_full_res")
    return value === "Only masked" || value === 1
  })
  const inverted = useStore(store, (s) => {
    const value = core(s, "inpainting_mask_invert")
    return value === "Inpaint not masked" || value === 1
  })

  return useMemo(() => {
    const values = valuesKey.split("|").map((v) => v || null)
    const editorLayers: Record<string, EditorLayer> = {}
    binding?.layers.forEach((layer, i) => {
      editorLayers[layer.param] = { kind: layer.kind, value: values[i + 1] ?? null, label: LAYER_LABELS[layer.param] }
    })
    const sized = width > 0 && height > 0
    const resizeBy = resizeTab === "Resize by" || resizeTab === 1
    return {
      image: values[0] ?? null,
      layers: editorLayers,
      activeLayer: layers.active?.param ?? null,
      error: errorsKey || null,
      // "Resize by": the frame (and its size label) follow the image × scale.
      frame: resizeBy ? { width: 0, height: 0, resizeMode: resizeMode ?? 1, scale: { by: scaleBy, step } } : sized ? { width, height, resizeMode: resizeMode ?? 1 } : null,
      inpaint: layers.mask ? { blur, padding, onlyMasked, inverted } : null,
      blankSize: sized ? { width, height } : null,
    }
  }, [valuesKey, binding, layers.active, layers.mask, errorsKey, width, height, resizeMode, resizeTab, scaleBy, step, blur, padding, onlyMasked, inverted])
}
