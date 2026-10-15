/**
 * Konva view of an editor document — the only file of the app that imports
 * Konva. Three layers (each its own <canvas>, DPR-aware): the background
 * image; the edited paint layer (its raster, plus the stroke in progress
 * drawn from the scratch raster with the brush opacity / eraser mode, cropped
 * to the stroke's bounds); overlays (selection shapes, the resize frame, the
 * inpaint area). Layer opacity and the mask-blur preview are CSS on the
 * paint layer's canvas, so they cost nothing per frame. Nothing listens to
 * Konva events: input is handled on the host element (interaction.ts).
 */
import Konva from "konva/lib/Core"
import { Image as KonvaImage } from "konva/lib/shapes/Image"
import { Line } from "konva/lib/shapes/Line"
import { Rect as KonvaRect } from "konva/lib/shapes/Rect"
import { Shape } from "konva/lib/Shape"
import type { Layer as KonvaLayer } from "konva/lib/Layer"
import type { Stage } from "konva/lib/Stage"
import type { AspectFrame } from "../core/overlays"
import type { Rect, View } from "../core/geometry"
import { isEmptyRect } from "../core/geometry"
import type { EditorDocument } from "./document"

export type Overlays = {
  /** The target frame of the resize mode (while the size sliders move). */
  frame?: AspectFrame | null
  /** The area "Only masked" inpaints. */
  area?: Rect | null
  /** A rectangle / lasso being drawn: flat image points. */
  shape?: number[] | null
  shapeErase?: boolean
}

export type LayerStyle = {
  /** Paint layer shown (and its opacity). */
  visible: boolean
  opacity: number
  /** Background image shown. */
  imageVisible: boolean
  /** Mask blur preview, image px (0 = off). */
  blur: number
}

const ACCENT = "#7c5cff"

export class Renderer {
  readonly element: HTMLDivElement
  private readonly doc: EditorDocument
  private readonly stage: Stage
  private readonly base: KonvaLayer
  private readonly paint: KonvaLayer
  private readonly overlay: KonvaLayer
  private readonly background: KonvaImage
  private readonly layerImage: KonvaImage
  private readonly preview: KonvaImage
  private readonly frame: Shape
  private readonly area: KonvaRect
  private readonly shape: Line
  private layerKey: string | null = null
  private view: View = { scale: 1, x: 0, y: 0 }
  private style: LayerStyle = { visible: true, opacity: 1, imageVisible: true, blur: 0 }
  private overlays: Overlays = {}
  private accent = ACCENT
  private readonly unsubscribe: () => void
  private dprQuery: MediaQueryList | null = null
  private readonly onDpr = () => this.updatePixelRatio()

  constructor(doc: EditorDocument) {
    this.doc = doc
    this.element = document.createElement("div")
    this.element.style.position = "absolute"
    this.element.style.inset = "0"
    Konva.pixelRatio = window.devicePixelRatio || 1
    this.stage = new Konva.Stage({ container: this.element, width: 1, height: 1, listening: false })
    this.base = new Konva.Layer({ listening: false })
    this.paint = new Konva.Layer({ listening: false })
    this.overlay = new Konva.Layer({ listening: false })
    this.stage.add(this.base, this.paint, this.overlay)

    this.background = new KonvaImage({ image: undefined, listening: false })
    this.base.add(this.background)
    this.layerImage = new KonvaImage({ image: undefined, listening: false })
    this.preview = new KonvaImage({ image: undefined, listening: false, visible: false })
    this.paint.add(this.layerImage, this.preview)

    this.frame = new Shape({ listening: false, visible: false, sceneFunc: (ctx) => this.drawFrame(ctx._context) })
    this.area = new KonvaRect({ listening: false, visible: false, strokeWidth: 1.5, dash: [6, 4], strokeScaleEnabled: false })
    this.shape = new Line({ listening: false, visible: false, closed: true, strokeWidth: 1.5, dash: [5, 4], strokeScaleEnabled: false })
    this.overlay.add(this.frame, this.area, this.shape)

    this.unsubscribe = doc.subscribe((event) => {
      if (event.type === "background") this.syncBackground()
      else if (event.type === "layer" && event.key === this.layerKey) this.syncLayer()
      else if (event.type === "preview" && event.key === this.layerKey) this.syncPreview()
    })
    this.syncBackground()
    this.watchPixelRatio()
  }

  /** Puts the stage in `host` (a positioned element). */
  attach(host: HTMLElement) {
    host.append(this.element)
  }

  setSize(width: number, height: number) {
    this.stage.size({ width: Math.max(1, Math.round(width)), height: Math.max(1, Math.round(height)) })
    this.stage.batchDraw()
  }

  setView(view: View) {
    this.view = view
    this.stage.scale({ x: view.scale, y: view.scale })
    this.stage.position({ x: view.x, y: view.y })
    // Pixels stay crisp when zoomed in (inspecting a mask edge), smooth when zoomed out.
    const smooth = view.scale < 2
    this.base.imageSmoothingEnabled(smooth)
    this.paint.imageSmoothingEnabled(smooth)
    this.applyStyle()
    this.stage.batchDraw()
  }

  /** Which document layer is shown (null: none). */
  setLayer(key: string | null) {
    if (key === this.layerKey) return
    this.layerKey = key
    this.syncLayer()
    this.syncPreview()
  }

  setStyle(style: LayerStyle) {
    this.style = style
    this.base.visible(style.imageVisible)
    this.applyStyle()
    this.stage.batchDraw()
  }

  setOverlays(overlays: Overlays) {
    this.overlays = overlays
    const { area, shape } = overlays
    this.frame.visible(Boolean(overlays.frame))
    this.area.visible(Boolean(area && !isEmptyRect(area)))
    if (area) this.area.setAttrs({ x: area.x, y: area.y, width: area.width, height: area.height, stroke: this.accent })
    this.shape.visible(Boolean(shape && shape.length >= 4))
    if (shape) {
      this.shape.setAttrs({
        points: shape,
        stroke: overlays.shapeErase ? "#ffffff" : this.accent,
        fill: overlays.shapeErase ? "rgba(255,255,255,0.12)" : `${this.accent}33`,
        shadowColor: "#000",
        shadowBlur: 2,
        shadowOpacity: 0.6,
        shadowForStrokeEnabled: true,
      })
    }
    this.overlay.batchDraw()
  }

  /** The brand colour for overlays (a CSS colour the canvas understands). */
  setAccent(color: string) {
    this.accent = color || ACCENT
    this.setOverlays(this.overlays)
  }

  redraw() {
    this.stage.batchDraw()
  }

  private applyStyle() {
    const canvas = this.paint.getNativeCanvasElement()
    canvas.style.opacity = String(this.style.visible ? this.style.opacity : 0)
    const blur = this.style.blur * this.view.scale
    canvas.style.filter = blur > 0.3 ? `blur(${blur.toFixed(2)}px)` : ""
  }

  private syncBackground() {
    const doc = this.doc
    this.background.setAttrs({ image: doc.background ?? undefined, x: 0, y: 0, width: doc.width, height: doc.height })
    this.syncLayer()
    this.syncPreview()
    this.stage.batchDraw()
  }

  private syncLayer() {
    const layer = this.layerKey ? this.doc.layer(this.layerKey) : null
    this.layerImage.setAttrs({ image: layer?.raster.source ?? undefined, x: 0, y: 0, width: this.doc.width, height: this.doc.height })
    this.paint.batchDraw()
  }

  private syncPreview() {
    const preview = this.doc.preview
    const scratch = this.doc.scratchRaster?.source
    if (!preview || preview.key !== this.layerKey || !scratch || isEmptyRect(preview.rect)) {
      this.preview.visible(false)
    } else {
      const { rect } = preview
      this.preview.setAttrs({
        visible: true,
        image: scratch,
        crop: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
        opacity: preview.opacity,
        globalCompositeOperation: preview.erase ? "destination-out" : "source-over",
      })
    }
    this.paint.batchDraw()
  }

  /** The resize frame: dims what the target leaves out (crop) or adds (fill), dashed outline. */
  private drawFrame(ctx: CanvasRenderingContext2D) {
    const frame = this.overlays.frame
    if (!frame) return
    const { width, height } = this.doc
    const { rect } = frame
    const px = 1 / this.view.scale
    ctx.save()
    ctx.beginPath()
    if (frame.kind === "crop") {
      ctx.rect(0, 0, width, height)
      ctx.rect(rect.x, rect.y, rect.width, rect.height)
      ctx.fillStyle = "rgba(0, 0, 0, 0.5)"
      ctx.fill("evenodd")
    } else if (frame.kind === "fill") {
      ctx.rect(rect.x, rect.y, rect.width, rect.height)
      ctx.rect(0, 0, width, height)
      ctx.fillStyle = "rgba(127, 127, 127, 0.35)"
      ctx.fill("evenodd")
    }
    ctx.lineWidth = 1.5 * px
    ctx.setLineDash([6 * px, 4 * px])
    ctx.strokeStyle = "rgba(0, 0, 0, 0.6)"
    ctx.strokeRect(rect.x, rect.y, rect.width, rect.height)
    ctx.lineDashOffset = 5 * px
    ctx.strokeStyle = "#ffffff"
    ctx.strokeRect(rect.x, rect.y, rect.width, rect.height)
    ctx.restore()
  }

  private watchPixelRatio() {
    this.dprQuery?.removeEventListener("change", this.onDpr)
    if (typeof window.matchMedia !== "function") return
    this.dprQuery = window.matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`)
    this.dprQuery.addEventListener("change", this.onDpr)
  }

  /** Moved to a screen with another density: re-render every canvas at the new ratio. */
  private updatePixelRatio() {
    const ratio = window.devicePixelRatio || 1
    Konva.pixelRatio = ratio
    for (const layer of [this.base, this.paint, this.overlay]) layer.getCanvas().setPixelRatio(ratio)
    this.applyStyle()
    this.stage.batchDraw()
    this.watchPixelRatio()
  }

  destroy() {
    this.unsubscribe()
    this.dprQuery?.removeEventListener("change", this.onDpr)
    this.stage.destroy()
    this.element.remove()
  }
}
