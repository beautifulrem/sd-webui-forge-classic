/**
 * What the processing will do with the image, for the editor's overlays
 * (pure): the target frame of the resize mode while the size sliders move
 * (`modules.images.resize_image`), and the "Only masked" inpaint area
 * (`modules.masking.get_crop_region_v2` + `expand_crop_region` after the mask
 * blur, `modules.processing` img2img init).
 */
import type { Rect, Size } from "./geometry"
import { isEmptyRect } from "./geometry"

/** `resize_mode` indices (remi.core.forms.RESIZE_MODES). */
export const RESIZE_MODES = ["Just Resize", "Crop and Resize", "Resize and Fill", "Latent Upscale", "Preserve Aspect Ratio"] as const

export type AspectFrame = {
  /** The part of the target the image covers, in image pixels (may extend past the image for "fill"). */
  rect: Rect
  /** stretch: the whole image is scaled to the target; crop: `rect` is kept; fill: the image sits inside `rect`, the rest is filled. */
  kind: "stretch" | "crop" | "fill" | "fit"
}

/** The frame the target size makes on the source image under a resize mode (an index or a mode name). */
export function aspectFrame(image: Size, target: Size, resizeMode: number | string): AspectFrame {
  const mode = typeof resizeMode === "number" ? resizeMode : Math.max(0, RESIZE_MODES.indexOf(resizeMode as (typeof RESIZE_MODES)[number]))
  const full: Rect = { x: 0, y: 0, width: image.width, height: image.height }
  if (image.width <= 0 || image.height <= 0 || target.width <= 0 || target.height <= 0) return { rect: full, kind: "stretch" }
  const ratio = target.width / target.height
  const source = image.width / image.height
  if (mode === 1) {
    // Crop and Resize: the largest centred rect of the target ratio.
    const width = ratio > source ? image.width : image.height * ratio
    const height = ratio > source ? image.width / ratio : image.height
    return { rect: centered(image, width, height), kind: "crop" }
  }
  if (mode === 2) {
    // Resize and Fill: the smallest centred rect of the target ratio around the image.
    const width = ratio < source ? image.width : image.height * ratio
    const height = ratio < source ? image.width / ratio : image.height
    return { rect: centered(image, width, height), kind: "fill" }
  }
  if (mode === 4) return { rect: full, kind: "fit" }
  return { rect: full, kind: "stretch" }
}

/** `remi.core.forms.snap_resolution`: the nearest multiple of `step` (at least one step). */
export function snapResolution(value: number, step: number): number {
  const unit = step > 0 ? step : 1
  return Math.max(unit, Math.floor(value / unit + 0.5) * unit)
}

/** The processing size of "Resize by": the image size × `by`, snapped like the server does (`modules.img2img`). */
export function scaledSize(image: Size, by: number, step: number): Size {
  return { width: snapResolution(image.width * by, step), height: snapResolution(image.height * by, step) }
}

function centered(image: Size, width: number, height: number): Rect {
  const w = Math.round(width)
  const h = Math.round(height)
  return { x: Math.round((image.width - w) / 2), y: Math.round((image.height - h) / 2), width: w, height: h }
}

/** How far the server's Gaussian mask blur reaches (half its kernel). */
export function blurReach(blur: number): number {
  return blur > 0 ? Math.trunc(2.5 * blur + 0.5) : 0
}

/** `get_crop_region_v2`: the mask bounds grown by `pad` and clipped (null without a mask). */
export function cropRegion(maskBounds: Rect, pad: number, image: Size): [number, number, number, number] | null {
  if (isEmptyRect(maskBounds)) return null
  const x1 = maskBounds.x
  const y1 = maskBounds.y
  const x2 = maskBounds.x + maskBounds.width
  const y2 = maskBounds.y + maskBounds.height
  if (!pad) return [x1, y1, x2, y2]
  return [Math.max(x1 - pad, 0), Math.max(y1 - pad, 0), Math.min(x2 + pad, image.width), Math.min(y2 + pad, image.height)]
}

const floorDiv = (a: number, b: number) => Math.floor(a / b)

/** `expand_crop_region`: grows a crop region to the processing ratio, kept inside the image. */
export function expandCropRegion(region: [number, number, number, number], processing: Size, image: Size): [number, number, number, number] {
  let [x1, y1, x2, y2] = region
  const ratioCrop = (x2 - x1) / (y2 - y1)
  const ratioProcessing = processing.width / processing.height
  if (ratioCrop > ratioProcessing) {
    const desired = (x2 - x1) / ratioProcessing
    const diff = Math.trunc(desired - (y2 - y1))
    y1 -= floorDiv(diff, 2)
    y2 += diff - floorDiv(diff, 2)
    if (y2 >= image.height) {
      const over = y2 - image.height
      y2 -= over
      y1 -= over
    }
    if (y1 < 0) {
      y2 -= y1
      y1 = 0
    }
    if (y2 >= image.height) y2 = image.height
  } else {
    const desired = (y2 - y1) * ratioProcessing
    const diff = Math.trunc(desired - (x2 - x1))
    x1 -= floorDiv(diff, 2)
    x2 += diff - floorDiv(diff, 2)
    if (x2 >= image.width) {
      const over = x2 - image.width
      x2 -= over
      x1 -= over
    }
    if (x1 < 0) {
      x2 -= x1
      x1 = 0
    }
    if (x2 >= image.width) x2 = image.width
  }
  return [x1, y1, x2, y2]
}

export type InpaintAreaInput = {
  /** Bounds of the painted mask (alpha > 0), image pixels. */
  maskBounds: Rect
  image: Size
  /** Processing size (core width/height). */
  target: Size
  blur: number
  padding: number
  /** "Inpaint not masked": the area is around everything but the mask. */
  inverted: boolean
}

/** The region "Only masked" inpaints, as a rect in image pixels (null: the whole picture is used). */
export function inpaintArea({ maskBounds, image, target, blur, padding, inverted }: InpaintAreaInput): Rect | null {
  if (isEmptyRect(maskBounds) || target.width <= 0 || target.height <= 0) return null
  const fullImage = maskBounds.width >= image.width && maskBounds.height >= image.height
  // An inverted mask covers everything outside the painted area: its bounds are the
  // whole image unless the paint covers whole rows or columns, which is rare.
  if (inverted && fullImage) return null
  const reach = blurReach(blur)
  const grown: Rect = inverted
    ? { x: 0, y: 0, width: image.width, height: image.height }
    : {
        x: Math.max(0, maskBounds.x - reach),
        y: Math.max(0, maskBounds.y - reach),
        width: Math.min(image.width, maskBounds.x + maskBounds.width + reach) - Math.max(0, maskBounds.x - reach),
        height: Math.min(image.height, maskBounds.y + maskBounds.height + reach) - Math.max(0, maskBounds.y - reach),
      }
  const crop = cropRegion(grown, padding, image)
  if (!crop) return null
  const [x1, y1, x2, y2] = expandCropRegion(crop, target, image)
  return { x: x1, y: y1, width: x2 - x1, height: y2 - y1 }
}
