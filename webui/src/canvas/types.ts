/** Public props of the image editor (no Konva here: importing types is free). */
import type { Size } from "./core/geometry"
import type { LayerKind } from "./core/modes"

export type { LayerKind } from "./core/modes"
export type { Size } from "./core/geometry"

export type EditorLayer = {
  kind: LayerKind
  /** The layer's upload id (null: empty). */
  value: string | null
  /** Shown in the layer list and the status line (English chrome key or already translated text). */
  label?: string
}

export type ImageEditorProps = {
  /**
   * Identity of the editor's document: layers and undo history survive
   * remounts under the same key (e.g. `"img2img"`, `"controlnet-0-input"`).
   */
  docKey: string
  /** The form id whose submit waits for this editor's uploads (`ui/pending-writes`). */
  pendingKey?: string
  /** Background image upload id. */
  image: string | null
  /** The user uploaded, dropped, pasted, created or removed the background. */
  onImageChange?: (image: { id: string; width: number; height: number } | null) => void
  /** Every layer the editor keeps (by param name), edited or not. */
  layers?: Record<string, EditorLayer>
  /** The layer the user edits now (a key of `layers`); null: view only. */
  activeLayer?: string | null
  /** A layer's new upload id (null: empty) after its edits were uploaded. */
  onLayerChange?: (key: string, id: string | null) => void

  /** No upload / replace / remove (the image comes from elsewhere). */
  noUpload?: boolean
  /** No painting at all (an image viewer with zoom and pan). */
  noScribble?: boolean
  /** Brushes paint in this colour; the colour picker is hidden. */
  fixedColor?: string
  /** A fixed brush size in px (the size control is hidden). */
  fixedSize?: number
  /** A fixed brush opacity, 0–1 (the opacity control is hidden). */
  fixedOpacity?: number
  /** A fixed brush softness, 0 (hard) – 1 (the softness control is hidden). */
  fixedSoftness?: number
  /** Offer a white canvas of this size when there is no image (sketching from scratch). */
  blankSize?: Size | null

  /**
   * Editor height in px; omitted: the "Canvas Height" setting
   * (`forge_canvas_height`), unless `fill` is set.
   */
  height?: number
  /** Fill the container instead (give it a height, e.g. a flex child); `height` is ignored. */
  fill?: boolean
  className?: string
  /** Accessible name of the editor. */
  label?: string
  /** Validation message to show (the param it belongs to was rejected). */
  error?: string | null
  /** Shown in the empty state under the upload prompt. */
  hint?: string

  /**
   * The processing size and resize mode: the target frame shows while they
   * change. `scale` ("Resize by") sizes the frame from the image instead:
   * image size × `scale.by`, snapped to multiples of `scale.step`
   * (`width` / `height` are ignored then).
   */
  frame?: { width: number; height: number; resizeMode: number | string; scale?: { by: number; step: number } | null } | null
  /** Inpaint options, for the mask blur and "Only masked" area previews. */
  inpaint?: { blur: number; padding: number; onlyMasked: boolean; inverted: boolean } | null
  /** "Use image size": receives the image size (the button shows only with this prop). */
  onUseImageSize?: (size: Size) => void
}
