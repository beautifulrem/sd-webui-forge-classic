/**
 * Pointer and wheel input of an editor view: painting (pointer events
 * coalesced, applied once per animation frame), shapes, fill, eyedropper,
 * pan (space / right or middle drag / hand tool / two fingers), zoom at the
 * cursor (wheel, Ctrl+wheel, pinch) and the brush cursor. Keyboard
 * shortcuts are handled by the editor component, which calls `setSpace` /
 * `setShift` here.
 */
import { constrainView, pinchView, type Point, screenToImage, type View, wheelZoomFactor, zoomAt } from "../core/geometry"
import type { LayerKind } from "../core/modes"
import type { Rgb } from "../core/pixels"
import type { StrokePoint } from "../core/stroke"
import { setPainting } from "../state/activity"
import type { BrushSettings, Tool } from "../state/tools"
import type { EditorDocument, PaintColor } from "./document"

export type InteractionContext = {
  /** The layer edits go to (null: view only). */
  layer: { key: string; kind: LayerKind } | null
  tool: Tool
  brush: BrushSettings
  eraser: BrushSettings
  /** Paint colour; null paints in the mask tint. */
  color: PaintColor
  fillTolerance: number
  fillSample: "layer" | "image"
  pressure: boolean
  /** Brush sizes are screen px (`forge_canvas_consistent_brush`) instead of image px. */
  screenBrush: boolean
  wheel: "zoom" | "pan"
  /** A W / A / S key held: the wheel adjusts size / opacity / softness. */
  adjust: "size" | "opacity" | "softness" | null
}

export type CursorState = { x: number; y: number; diameter: number; mode: "brush" | "eraser" | "cross" | "pick" | "grab" | "grabbing" | "default"; inside: boolean }

export type InteractionCallbacks = {
  context: () => InteractionContext
  view: () => View
  viewport: () => { width: number; height: number }
  setView: (view: View) => void
  /** The eyedropper sampled a colour (`final` on release). */
  picked: (color: Rgb, final: boolean) => void
  cursor: (state: CursorState | null) => void
  shape: (points: number[] | null, erase: boolean) => void
  /** Wheel with W / A / S held: one notch up (+1) or down (−1). */
  adjust: (what: "size" | "opacity" | "softness", direction: 1 | -1) => void
  /** A paint edit was committed (the colour it used, for the recent list). */
  painted: (color: PaintColor) => void
}

type Gesture =
  | { type: "stroke"; pointer: number; erase: boolean }
  | { type: "pan"; pointer: number; last: Point }
  | { type: "pinch"; start: View; from: [Point, Point]; ids: [number, number] }
  | { type: "rect"; pointer: number; start: Point; end: Point; erase: boolean }
  | { type: "lasso"; pointer: number; points: number[]; lastScreen: Point; erase: boolean }
  | { type: "pick"; pointer: number }

export class Interaction {
  private readonly host: HTMLElement
  private readonly doc: EditorDocument
  private readonly cb: InteractionCallbacks
  private current: Gesture | null = null
  private readonly pointers = new Map<number, Point>()
  private queue: StrokePoint[] = []
  private frame = 0
  private hover: Point | null = null
  private space = false
  private shift = false
  private bounds: DOMRect | null = null
  private readonly abort = new AbortController()

  private get gesture(): Gesture | null {
    return this.current
  }

  /** Every gesture change goes through here: painting ones are announced (`state/activity`). */
  private set gesture(value: Gesture | null) {
    this.current = value
    setPainting(this, value !== null && value.type !== "pan" && value.type !== "pinch")
  }

  constructor(host: HTMLElement, doc: EditorDocument, callbacks: InteractionCallbacks) {
    this.host = host
    this.doc = doc
    this.cb = callbacks
    const options = { signal: this.abort.signal }
    host.addEventListener("pointerdown", this.onDown, options)
    host.addEventListener("pointermove", this.onMove, options)
    host.addEventListener("pointerup", this.onUp, options)
    host.addEventListener("pointercancel", this.onCancel, options)
    host.addEventListener("pointerleave", this.onLeave, options)
    host.addEventListener("wheel", this.onWheel, { ...options, passive: false })
    host.addEventListener("contextmenu", (event) => event.preventDefault(), options)
    // Safari's own pinch gestures would zoom the page.
    host.addEventListener("gesturestart", (event) => event.preventDefault(), options)
  }

  /**
   * Detaches from the host. A stroke in progress is committed with the
   * points it got so far (the editor can unmount mid-stroke, e.g. when the
   * stage switches view): the user's paint is never thrown away.
   */
  destroy() {
    this.abort.abort()
    cancelAnimationFrame(this.frame)
    this.frame = 0
    const gesture = this.gesture
    this.gesture = null
    if (gesture?.type === "stroke" && this.doc.stroking) {
      const points = this.queue
      this.queue = []
      this.doc.extendStroke(points)
      this.doc.endStroke()
      if (!gesture.erase) this.cb.painted(this.cb.context().color)
    }
    this.queue = []
  }

  /** Space held: dragging pans. */
  setSpace(held: boolean) {
    if (this.space === held) return
    this.space = held
    this.updateCursor()
  }

  /** Shift held: brushes erase, shapes subtract. */
  setShift(held: boolean) {
    if (this.shift === held) return
    this.shift = held
    this.updateCursor()
  }

  get busy() {
    return this.gesture !== null
  }

  // ---------------------------------------------------------------- helpers

  private local(event: { clientX: number; clientY: number }): Point {
    const rect = (this.bounds ??= this.host.getBoundingClientRect())
    return { x: event.clientX - rect.left, y: event.clientY - rect.top }
  }

  private toImage(point: Point): Point {
    return screenToImage(this.cb.view(), point)
  }

  /** Effective tool: Space / hand pans; Shift turns the brush into the eraser. */
  private effectiveTool(ctx: InteractionContext): Tool {
    if (this.space || ctx.tool === "hand" || !ctx.layer) return "hand"
    if (ctx.tool === "brush" && this.shift) return "eraser"
    return ctx.tool
  }

  private brushFor(ctx: InteractionContext, tool: Tool): BrushSettings {
    return tool === "eraser" ? ctx.eraser : ctx.brush
  }

  /** Brush diameter in image px. */
  private imageSize(ctx: InteractionContext, brush: BrushSettings): number {
    return ctx.screenBrush ? brush.size / this.cb.view().scale : brush.size
  }

  private schedule() {
    if (this.frame) return
    this.frame = requestAnimationFrame(() => {
      this.frame = 0
      this.flush()
    })
  }

  /** Applies queued stroke points and the cursor once per frame. */
  private flush() {
    if (this.queue.length > 0 && this.gesture?.type === "stroke") {
      const points = this.queue
      this.queue = []
      this.doc.extendStroke(points)
    }
    this.updateCursor()
  }

  private updateCursor() {
    const ctx = this.cb.context()
    const tool = this.effectiveTool(ctx)
    const point = this.hover
    if (!point) {
      this.cb.cursor(null)
      return
    }
    const image = this.toImage(point)
    const inside = image.x >= 0 && image.y >= 0 && image.x < this.doc.width && image.y < this.doc.height
    let mode: CursorState["mode"] = "default"
    let diameter = 0
    if (tool === "hand") mode = this.gesture?.type === "pan" ? "grabbing" : ctx.layer || this.space || ctx.tool === "hand" ? "grab" : "default"
    else if (tool === "brush" || tool === "eraser") {
      mode = tool
      diameter = this.imageSize(ctx, this.brushFor(ctx, tool)) * this.cb.view().scale
    } else if (tool === "eyedropper") mode = "pick"
    else mode = "cross"
    if (!this.doc.hasImage) mode = "default"
    this.cb.cursor({ x: point.x, y: point.y, diameter, mode, inside })
  }

  private setView(view: View) {
    this.cb.setView(constrainView(view, { width: this.doc.width, height: this.doc.height }, this.cb.viewport()))
  }

  // ---------------------------------------------------------------- pointer events

  private readonly onDown = (event: PointerEvent) => {
    if (!this.doc.hasImage) return
    this.bounds = this.host.getBoundingClientRect()
    const point = this.local(event)
    this.pointers.set(event.pointerId, point)
    this.hover = point

    // A second finger: pinch / pan the view; a stroke the first finger started is dropped.
    if (event.pointerType === "touch" && this.pointers.size === 2) {
      if (this.gesture?.type === "stroke") this.doc.cancelStroke()
      if (this.gesture?.type === "rect" || this.gesture?.type === "lasso") this.cb.shape(null, false)
      this.queue = []
      const [a, b] = [...this.pointers.entries()]
      this.gesture = { type: "pinch", start: this.cb.view(), from: [a![1], b![1]], ids: [a![0], b![0]] }
      return
    }
    if (this.gesture) return

    const ctx = this.cb.context()
    // Nothing to paint: one finger is the page's (it scrolls); two fingers pan and zoom.
    if (event.pointerType === "touch" && !ctx.layer && !this.space) return
    const tool = this.effectiveTool(ctx)
    const pan = event.button === 1 || event.button === 2 || tool === "hand"
    if (event.button !== 0 && !pan) return
    event.preventDefault()
    this.host.setPointerCapture(event.pointerId)

    if (pan) {
      this.gesture = { type: "pan", pointer: event.pointerId, last: point }
      this.updateCursor()
      return
    }
    const layer = ctx.layer!
    const image = this.toImage(point)
    const erase = this.shift || tool === "eraser"
    switch (tool) {
      case "brush":
      case "eraser": {
        const brush = this.brushFor(ctx, tool)
        this.doc.beginStroke(layer.key, layer.kind, {
          size: this.imageSize(ctx, brush),
          hardness: 1 - brush.softness,
          // Masks are binary to the processing (thresholded at 50 %): always paint them at full strength.
          opacity: layer.kind === "mask" ? 1 : brush.opacity,
          color: ctx.color,
          erase: tool === "eraser",
          pressure: ctx.pressure && event.pointerType === "pen",
        })
        this.gesture = { type: "stroke", pointer: event.pointerId, erase: tool === "eraser" }
        this.queue.push({ ...image, pressure: event.pointerType === "pen" ? event.pressure || 0.5 : 1 })
        this.schedule()
        return
      }
      case "rect":
        this.gesture = { type: "rect", pointer: event.pointerId, start: image, end: image, erase }
        return
      case "lasso":
        this.gesture = { type: "lasso", pointer: event.pointerId, points: [image.x, image.y], lastScreen: point, erase }
        return
      case "fill": {
        const done = this.doc.fill(layer.key, layer.kind, image.x, image.y, ctx.fillTolerance, ctx.fillSample, {
          color: ctx.color,
          opacity: layer.kind === "mask" ? 1 : ctx.brush.opacity,
          erase,
        })
        if (done && !erase) this.cb.painted(ctx.color)
        this.host.releasePointerCapture(event.pointerId)
        return
      }
      case "eyedropper": {
        this.gesture = { type: "pick", pointer: event.pointerId }
        const color = this.doc.sample(image.x, image.y, layer.kind === "sketch" ? layer.key : null)
        if (color) this.cb.picked(color, false)
        return
      }
    }
  }

  private readonly onMove = (event: PointerEvent) => {
    const point = this.local(event)
    this.hover = point
    if (this.pointers.has(event.pointerId)) this.pointers.set(event.pointerId, point)
    const gesture = this.gesture
    if (!gesture) {
      this.schedule()
      return
    }
    switch (gesture.type) {
      case "pinch": {
        const a = this.pointers.get(gesture.ids[0])
        const b = this.pointers.get(gesture.ids[1])
        if (a && b) this.setView(pinchView(gesture.start, gesture.from, [a, b]))
        break
      }
      case "pan": {
        if (event.pointerId !== gesture.pointer) break
        const view = this.cb.view()
        this.setView({ ...view, x: view.x + point.x - gesture.last.x, y: view.y + point.y - gesture.last.y })
        gesture.last = point
        break
      }
      case "stroke": {
        if (event.pointerId !== gesture.pointer) break
        const events = typeof event.getCoalescedEvents === "function" ? event.getCoalescedEvents() : []
        for (const e of events.length > 0 ? events : [event]) {
          const p = this.toImage(this.local(e))
          this.queue.push({ ...p, pressure: e.pointerType === "pen" ? e.pressure || 0.5 : 1 })
        }
        break
      }
      case "rect": {
        if (event.pointerId !== gesture.pointer) break
        gesture.end = this.toImage(point)
        const { start, end } = gesture
        this.cb.shape([start.x, start.y, end.x, start.y, end.x, end.y, start.x, end.y], gesture.erase)
        break
      }
      case "lasso": {
        if (event.pointerId !== gesture.pointer) break
        // A point every 2 screen px keeps the polygon light.
        if (Math.hypot(point.x - gesture.lastScreen.x, point.y - gesture.lastScreen.y) < 2) break
        const p = this.toImage(point)
        gesture.points.push(p.x, p.y)
        gesture.lastScreen = point
        this.cb.shape(gesture.points, gesture.erase)
        break
      }
      case "pick": {
        if (event.pointerId !== gesture.pointer) break
        const p = this.toImage(point)
        const ctx = this.cb.context()
        const color = this.doc.sample(p.x, p.y, ctx.layer?.kind === "sketch" ? ctx.layer.key : null)
        if (color) this.cb.picked(color, false)
        break
      }
    }
    this.schedule()
  }

  private readonly onUp = (event: PointerEvent) => {
    this.pointers.delete(event.pointerId)
    const gesture = this.gesture
    if (!gesture) return
    if (gesture.type === "pinch") {
      // Lifting one finger ends the pinch; the other does nothing until it lifts too.
      if (this.pointers.size === 0) this.gesture = null
      return
    }
    if (gesture.pointer !== event.pointerId) return
    this.gesture = null
    const ctx = this.cb.context()
    switch (gesture.type) {
      case "stroke": {
        cancelAnimationFrame(this.frame)
        this.frame = 0
        const points = this.queue
        this.queue = []
        this.doc.extendStroke(points)
        this.doc.endStroke()
        if (!gesture.erase) this.cb.painted(ctx.color)
        break
      }
      case "rect":
      case "lasso": {
        this.cb.shape(null, false)
        const points = gesture.type === "rect" ? rectPoints(gesture.start, gesture.end) : gesture.points
        if (ctx.layer && points.length >= 6 && area(points) >= 1) {
          this.doc.polygon(ctx.layer.key, ctx.layer.kind, points, { color: ctx.color, opacity: ctx.layer.kind === "mask" ? 1 : ctx.brush.opacity, erase: gesture.erase })
          if (!gesture.erase) this.cb.painted(ctx.color)
        }
        break
      }
      case "pick": {
        const p = this.toImage(this.local(event))
        const color = this.doc.sample(p.x, p.y, ctx.layer?.kind === "sketch" ? ctx.layer.key : null)
        if (color) this.cb.picked(color, true)
        break
      }
    }
    if (this.host.hasPointerCapture(event.pointerId)) this.host.releasePointerCapture(event.pointerId)
    this.updateCursor()
  }

  private readonly onCancel = (event: PointerEvent) => {
    this.pointers.delete(event.pointerId)
    const gesture = this.gesture
    if (!gesture) return
    if (gesture.type === "pinch") {
      if (this.pointers.size === 0) this.gesture = null
      return
    }
    if (gesture.pointer !== event.pointerId) return
    this.gesture = null
    this.queue = []
    if (gesture.type === "stroke") this.doc.cancelStroke()
    if (gesture.type === "rect" || gesture.type === "lasso") this.cb.shape(null, false)
  }

  private readonly onLeave = (event: PointerEvent) => {
    if (this.gesture && "pointer" in this.gesture && this.gesture.pointer === event.pointerId) return
    this.hover = null
    this.bounds = null
    this.cb.cursor(null)
  }

  private readonly onWheel = (event: WheelEvent) => {
    if (!this.doc.hasImage) return
    event.preventDefault()
    this.bounds = this.host.getBoundingClientRect()
    const ctx = this.cb.context()
    const direction = event.deltaY < 0 ? 1 : -1
    if (ctx.adjust && event.deltaY !== 0) {
      this.cb.adjust(ctx.adjust, direction)
      this.updateCursor()
      return
    }
    const point = this.local(event)
    const view = this.cb.view()
    // Trackpads scroll in both axes and pinch as Ctrl+wheel; mouse wheels zoom (or pan, by preference).
    const trackpadScroll = !event.ctrlKey && event.deltaMode === 0 && event.deltaX !== 0
    if (event.ctrlKey || event.metaKey || (ctx.wheel === "zoom" && !trackpadScroll && !event.shiftKey)) {
      this.setView(zoomAt(view, wheelZoomFactor(event.deltaY, event.deltaMode), point))
    } else {
      const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 400 : 1
      const dx = (event.shiftKey && event.deltaX === 0 ? event.deltaY : event.deltaX) * unit
      const dy = (event.shiftKey && event.deltaX === 0 ? 0 : event.deltaY) * unit
      this.setView({ ...view, x: view.x - dx, y: view.y - dy })
    }
    this.hover = point
    this.updateCursor()
  }
}

function rectPoints(a: Point, b: Point): number[] {
  return [a.x, a.y, b.x, a.y, b.x, b.y, a.x, b.y]
}

/** Polygon area (shoelace), image px². */
function area(points: readonly number[]): number {
  let sum = 0
  const n = points.length >> 1
  for (let i = 0, j = n - 1; i < n; j = i++) sum += (points[j * 2]! + points[i * 2]!) * (points[j * 2 + 1]! - points[i * 2 + 1]!)
  return Math.abs(sum / 2)
}
