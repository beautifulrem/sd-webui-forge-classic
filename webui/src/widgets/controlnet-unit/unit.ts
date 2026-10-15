/**
 * Pure rules of a ControlNet unit value (the JSON of one `controlnet-unit`
 * param). Mirrors `lib_controlnet/unit_param.py` (defaults, the paste marker)
 * and the old Gradio wiring (control type → preprocessor / model defaults,
 * preprocessor → slider values, edits that drop the preview);
 * `contract.json` is the server's export these tests compare against.
 */

export type ImageKey = "image" | "sketch" | "image_fg" | "mask_image" | "mask_image_fg" | "generated_image"

export type UnitValue = {
  enabled: boolean
  type_filter: string
  module: string
  model: string
  weight: number
  guidance_start: number
  guidance_end: number
  processor_res: number
  threshold_a: number
  threshold_b: number
  pixel_perfect: boolean
  control_mode: string
  resize_mode: string
  hr_option: string
  save_detected_map: boolean
  use_preview_as_input: boolean
  image: string | null
  sketch: string | null
  image_fg: string | null
  mask_image: string | null
  mask_image_fg: string | null
  generated_image: string | null
  mask_upload: boolean
  pose: unknown
  /** img2img only: read the unit's own image instead of the img2img image. */
  upload_independent?: boolean
}

export type SliderSpec = { label?: string; minimum?: number; maximum?: number; step?: number; value?: number; visible: boolean }

export type PreprocessorInfo = {
  name: string
  tags: string[]
  resolution: SliderSpec
  slider_1: SliderSpec
  slider_2: SliderSpec
  /** Whether "Control Mode" applies. */
  control_mode: boolean
  /** Whether a ControlNet model is needed. */
  model: boolean
}

export type ControlType = { name: string; modules: string[]; models: string[]; default_module: string; default_model: string }

export type Catalog = { preprocessors: PreprocessorInfo[]; models: string[]; types: ControlType[] }

export type Control = { label: string; info?: string; minimum?: number; maximum?: number; step?: number; default?: number | string; choices?: string[] }

export type UnitProps = {
  unit: number
  img2img: boolean
  /** Titles of the tab strip holding the units (to switch to another unit). */
  tabs: string[]
  controls: Record<string, Control>
  invert_hint: string
  catalog: Catalog
}

export const PASTE_MARKER = "__paste__"

export const DEFAULT_UNIT: UnitValue = {
  enabled: false,
  type_filter: "All",
  module: "None",
  model: "None",
  weight: 1,
  guidance_start: 0,
  guidance_end: 1,
  processor_res: -1,
  threshold_a: -1,
  threshold_b: -1,
  pixel_perfect: false,
  control_mode: "Balanced",
  resize_mode: "Crop and Resize",
  hr_option: "Both",
  save_detected_map: true,
  use_preview_as_input: false,
  image: null,
  sketch: null,
  image_fg: null,
  mask_image: null,
  mask_image_fg: null,
  generated_image: null,
  mask_upload: false,
  pose: null,
}

export function defaultUnit(img2img: boolean): UnitValue {
  return img2img ? { ...DEFAULT_UNIT, upload_independent: false } : { ...DEFAULT_UNIT }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

/** The unit for any stored value: known keys of the right type over the defaults (the server re-checks). */
export function normalizeUnit(value: unknown, img2img: boolean): UnitValue {
  const unit = defaultUnit(img2img)
  if (!isRecord(value)) return unit
  const target = unit as Record<string, unknown>
  for (const key of Object.keys(target)) {
    if (!(key in value)) continue
    const raw = value[key]
    if (key === "pose") target[key] = raw ?? null
    else if (isImageKey(key)) target[key] = typeof raw === "string" && raw !== "" ? raw : null
    else if (raw !== null && typeof raw === typeof target[key]) target[key] = raw
  }
  return unit
}

const IMAGE_KEYS = new Set<string>(["image", "sketch", "image_fg", "mask_image", "mask_image_fg", "generated_image"])
const isImageKey = (key: string) => IMAGE_KEYS.has(key)

/** A pasted (partial) unit: the fields an infotext records, marked by the server. */
export function isPasted(value: unknown): value is Record<string, unknown> {
  return isRecord(value) && value[PASTE_MARKER] === true
}

// The old UI dropped the preview when one of these changed (register_clear_preview).
const PREVIEW_INPUTS: readonly (keyof UnitValue)[] = ["pixel_perfect", "module", "image", "sketch", "processor_res", "threshold_a", "threshold_b", "upload_independent"]

/** `unit` with `patch` applied; a change of what the preview was made from drops it. */
export function applyEdit(unit: UnitValue, patch: Partial<UnitValue>): UnitValue {
  const next = { ...unit, ...patch }
  const stale = PREVIEW_INPUTS.some((key) => key in patch && !("generated_image" in patch) && patch[key] !== unit[key])
  return stale ? { ...next, generated_image: null, use_preview_as_input: false, pose: null } : next
}

export function preprocessorOf(catalog: Catalog, name: string): PreprocessorInfo | undefined {
  return catalog.preprocessors.find((p) => p.name === name)
}

export function typeOf(catalog: Catalog, name: string): ControlType | undefined {
  return catalog.types.find((t) => t.name === name) ?? catalog.types.find((t) => t.name === "All")
}

/** Preprocessors the control type offers (the current one stays listed). */
export function modulesFor(unit: UnitValue, catalog: Catalog): string[] {
  const list = typeOf(catalog, unit.type_filter)?.modules ?? catalog.preprocessors.map((p) => p.name)
  return list.includes(unit.module) ? list : [...list, unit.module]
}

/** Models the control type offers (the current one stays listed). */
export function modelsFor(unit: UnitValue, catalog: Catalog): string[] {
  const list = typeOf(catalog, unit.type_filter)?.models ?? catalog.models
  return list.includes(unit.model) ? list : [...list, unit.model]
}

/** The values a preprocessor's sliders start at (the old `build_sliders` updates set them). */
function sliderValues(preprocessor: PreprocessorInfo | undefined): Partial<UnitValue> {
  if (!preprocessor) return {}
  const patch: Partial<UnitValue> = {}
  if (typeof preprocessor.resolution.value === "number") patch.processor_res = preprocessor.resolution.value
  if (typeof preprocessor.slider_1.value === "number") patch.threshold_a = preprocessor.slider_1.value
  if (typeof preprocessor.slider_2.value === "number") patch.threshold_b = preprocessor.slider_2.value
  return patch
}

/** A new preprocessor, with its sliders' default values. */
export function selectModule(unit: UnitValue, name: string, catalog: Catalog): UnitValue {
  return applyEdit(unit, { module: name, ...sliderValues(preprocessorOf(catalog, name)) })
}

/** A new control type: its default preprocessor and model, as the old radio did. */
export function selectType(unit: UnitValue, name: string, catalog: Catalog): UnitValue {
  const type = typeOf(catalog, name)
  if (!type) return applyEdit(unit, { type_filter: name })
  return { ...selectModule({ ...unit, type_filter: name }, type.default_module, catalog), model: type.default_model }
}

/** `pasted` (a paste's partial unit) over `current`; images, "Enable" and the rest are kept. */
export function mergePasted(current: UnitValue, pasted: Record<string, unknown>, catalog?: Catalog | null): UnitValue {
  const fields = normalizeUnit({ ...current, ...pasted }, "upload_independent" in current)
  const patch: Partial<UnitValue> = {}
  for (const key of Object.keys(pasted)) {
    if (key !== PASTE_MARKER && key in fields) (patch as Record<string, unknown>)[key] = (fields as Record<string, unknown>)[key]
  }
  const next = applyEdit(current, patch)
  if (catalog && !modulesListed(next, catalog)) next.type_filter = "All"
  return next
}

function modulesListed(unit: UnitValue, catalog: Catalog): boolean {
  return (typeOf(catalog, unit.type_filter)?.modules ?? []).includes(unit.module)
}

/** A slider's shown value: the unit's, or the preprocessor's default for an unset (-1) one. */
export function sliderValue(value: number, spec: SliderSpec): number {
  return value < 0 && typeof spec.value === "number" ? spec.value : value
}

/** The old "send dimensions" rounding: the closest multiple of 8. */
export function closestEight(size: number): number {
  const rest = size % 8
  return rest <= 4 ? Math.round(size - rest) : Math.round(size + (8 - rest))
}

/** Whether the unit reads its own images (txt2img, or img2img with an independent image). */
export function ownsImage(unit: UnitValue, img2img: boolean): boolean {
  return !img2img || unit.upload_independent === true
}
