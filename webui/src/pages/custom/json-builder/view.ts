/**
 * The canvas view's maths (upstream `main.js` fitStage / zoomAt / clampPan):
 * the stage is fitted into the canvas area at the draft's aspect ratio, then
 * zoomed about a point and panned, never further than its overflow.
 */
import { clamp } from "./draft"

export type View = { scale: number; panX: number; panY: number }

export const MIN_ZOOM = 0.25
export const MAX_ZOOM = 8
/** One wheel notch. */
export const ZOOM_STEP = 1.2
export const RESET_VIEW: View = { scale: 1, panX: 0, panY: 0 }

/** The largest `ratio` (width / height) rectangle inside `width` × `height`. */
export function fitStage(width: number, height: number, ratio: number): { width: number; height: number } {
  const cw = Math.max(1, width)
  const ch = Math.max(1, height)
  return cw / ratio > ch ? { width: ch * ratio, height: ch } : { width: cw, height: cw / ratio }
}

/** A pan along one axis: a stage larger than the area may move until its edge, a smaller one within the area. */
export function clampAxis(pan: number, areaSize: number, stageSize: number): number {
  const diff = Math.abs(stageSize - areaSize) / 2
  return clamp(pan, -diff, diff) + 0 // no -0 in the transform
}

/** `view` with its pan kept within bounds, for a stage of `stage` (unscaled) in an area of `area`. */
export function clampView(view: View, area: { width: number; height: number }, stage: { width: number; height: number }): View {
  return { ...view, panX: clampAxis(view.panX, area.width, stage.width * view.scale), panY: clampAxis(view.panY, area.height, stage.height * view.scale) }
}

/**
 * `view` zoomed by `factor` about a point (`x`, `y` from the area's centre),
 * which stays under the pointer; the zoom stays within {@link MIN_ZOOM}–{@link MAX_ZOOM}.
 */
export function zoomAbout(view: View, x: number, y: number, factor: number): View {
  const scale = clamp(view.scale * factor, MIN_ZOOM, MAX_ZOOM)
  const ratio = scale / view.scale
  if (ratio === 1) return view
  return { scale, panX: view.panX + (x - view.panX) * (1 - ratio), panY: view.panY + (y - view.panY) * (1 - ratio) }
}
