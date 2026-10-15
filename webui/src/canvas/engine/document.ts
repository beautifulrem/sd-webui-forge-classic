/**
 * The editor document: a background image and its paint layers (one raster
 * each, at the image's resolution), the undo history and each layer's
 * upload sync. Documents live in a registry by key, so the layers and the
 * history survive the editor being unmounted (switching pages, maximising)
 * and are only dropped when the background changes.
 *
 * Every edit is "draw into the scratch raster, composite onto the layer":
 * the history keeps the layer's pixels under the edited rectangle before the
 * edit (a compact patch) and the edit itself (dabs, a polygon, a fill
 * region), so undo writes the patch back and redo replays the edit.
 */
import { clipRect, EMPTY_RECT, isEmptyRect, type Rect, rectFromBounds, type Size, unionRect } from "../core/geometry"
import { History, type HistoryLimits } from "../core/history"
import { decodePatch, encodePatch, type Patch, type PatchFormat, patchBytes } from "../core/patch"
import { alphaBounds, hasContent, maskAlphaFromImage, type Rgb, tintAlpha, toMaskExport } from "../core/pixels"
import type { LayerKind } from "../core/modes"
import { floodFill } from "../core/flood-fill"
import { tilesForDabs, tilesInRect } from "../core/tiles"
import { dabBounds, type DabSpec, type StrokePoint, StrokeWalker } from "../core/stroke"
import type { Raster, RasterFactory } from "./raster"
import { createUploadCache, LayerSync, type SyncDeps, type UploadCache } from "./sync"

/** Pixels of a layer in wire format. */
export type LayerExport = { data: Uint8ClampedArray; width: number; height: number } | null

/** A colour, or `null` for a mask layer's display tint. */
export type PaintColor = Rgb | null

type Draw =
  | { type: "dabs"; dabs: Float32Array; hardness: number; color: PaintColor }
  | { type: "polygon"; points: Float32Array; color: PaintColor }
  | { type: "region"; rect: Rect; alpha: Patch; color: PaintColor }

type Op = { type: "paint"; draw: Draw; opacity: number; erase: boolean } | { type: "invert" } | { type: "clear" }

/** The layer's pixels before an edit, per tile. */
type Snapshot = { rect: Rect; patch: Patch }[]

/** `before` is null for edits that are their own inverse (invert): undo applies them again. */
type Entry = { layer: string; rect: Rect; before: Snapshot | null; op: Op; bytes: number }

export type Layer = {
  readonly key: string
  readonly kind: LayerKind
  readonly raster: Raster
  /** Bumped by every content change. */
  version: number
  /** What is known about the pixels without reading them (exports skip the read when known). */
  content: "empty" | "painted" | "unknown"
  /**
   * Where the painted pixels are, kept up to date by the edits so the "Only
   * masked" preview never reads the whole layer: `exact` (to the pixel or two
   * of a brush's margin), an upper bound (`exact: false`: only that rectangle
   * is scanned when asked), or null (unknown: the whole layer is scanned).
   */
  bounds: { rect: Rect; exact: boolean } | null
  readonly sync: LayerSync
}

export type DocEvent =
  | { type: "background" }
  | { type: "layer"; key: string; origin: "user" | "external" }
  | { type: "preview"; key: string }
  | { type: "history" }

export type StrokeOptions = {
  size: number
  hardness: number
  opacity: number
  color: PaintColor
  erase: boolean
  pressure: boolean
  spacing?: number
}

export type PaintOptions = { color: PaintColor; opacity: number; erase: boolean }

export type Preview = { key: string; rect: Rect; opacity: number; erase: boolean }

export type DocumentDeps = {
  createRaster: RasterFactory
  /** Upload plumbing of layer syncs: PNG encoding of raw pixels (rasters without `toPng`) and the upload. */
  sync: Omit<SyncDeps, "png" | "cache" | "busy"> & { encode: (data: Uint8ClampedArray, width: number, height: number) => Promise<Blob> }
  history?: HistoryLimits
}

const formatOf = (kind: LayerKind): PatchFormat => (kind === "mask" ? "alpha" : "rgba")

export class EditorDocument {
  readonly deps: DocumentDeps
  private size: Size = { width: 0, height: 0 }
  /** The background as drawn (decoded image); null until loaded. */
  private image: CanvasImageSource | null = null
  /** The background's upload id. */
  imageId: string | null = null
  private sampleRaster: Raster | null = null
  private readonly layers = new Map<string, Layer>()
  private scratch: Raster | null = null
  private readonly history: History<Entry>
  private readonly listeners = new Set<(event: DocEvent) => void>()
  private readonly cache: UploadCache = createUploadCache()
  private stroke: { key: string; walker: StrokeWalker; options: StrokeOptions; dabs: number[]; rect: Rect } | null = null
  /** Display colour of mask layers. */
  tint: Rgb = [255, 59, 107]
  /** Changes with every event (useSyncExternalStore snapshot). */
  revision = 0
  /** Receives a layer's new upload id (null: empty) once its edits are uploaded. */
  onCommit: ((key: string, id: string | null) => void) | null = null
  onSyncError: ((error: unknown) => void) | null = null

  constructor(deps: DocumentDeps) {
    this.deps = deps
    this.history = new History<Entry>(deps.history)
  }

  // ------------------------------------------------------------ events

  subscribe(listener: (event: DocEvent) => void): () => void {
    this.listeners.add(listener)
    return () => void this.listeners.delete(listener)
  }

  private emit(event: DocEvent) {
    this.revision++
    for (const listener of this.listeners) listener(event)
  }

  // ------------------------------------------------------------ background

  get width() {
    return this.size.width
  }

  get height() {
    return this.size.height
  }

  get background(): CanvasImageSource | null {
    return this.image
  }

  get hasImage() {
    return this.size.width > 0 && this.size.height > 0
  }

  /**
   * Sets the background. A different size drops every layer; `keepLayers`
   * false (a new picture) clears them and the history too.
   */
  setBackground(image: CanvasImageSource | null, size: Size, id: string | null, { keepLayers = false }: { keepLayers?: boolean } = {}) {
    const resized = size.width !== this.size.width || size.height !== this.size.height
    this.cancelStroke()
    this.image = image
    this.imageId = id
    this.sampleRaster = null
    if (resized) {
      for (const layer of this.layers.values()) {
        layer.sync.adopt(null)
        layer.sync.dispose()
        disposeRaster(layer.raster)
      }
      this.layers.clear()
      if (this.scratch) disposeRaster(this.scratch)
      this.scratch = null
      this.size = { ...size }
    }
    if (resized || !keepLayers) {
      for (const layer of this.layers.values()) {
        layer.raster.clear()
        layer.version++
        layer.content = "empty"
        layer.bounds = { rect: EMPTY_RECT, exact: true }
        layer.sync.adopt(null)
      }
      this.history.clear()
    }
    this.emit({ type: "background" })
    this.emit({ type: "history" })
  }

  // ------------------------------------------------------------ layers

  layer(key: string): Layer | null {
    return this.layers.get(key) ?? null
  }

  /** The layer `key`, created empty on first use. */
  ensureLayer(key: string, kind: LayerKind): Layer {
    const existing = this.layers.get(key)
    if (existing) return existing
    if (!this.hasImage) throw new Error("no background image")
    const raster = this.deps.createRaster(this.size.width, this.size.height)
    const sync = new LayerSync({ ...this.deps.sync, png: () => this.exportPng(key), cache: this.cache, busy: () => this.stroke !== null })
    sync.bind(
      (id) => this.onCommit?.(key, id),
      (error) => this.onSyncError?.(error),
    )
    const layer: Layer = { key, kind, raster, version: 0, content: "empty", bounds: { rect: EMPTY_RECT, exact: true }, sync }
    this.layers.set(key, layer)
    return layer
  }

  private changed(layer: Layer, origin: "user" | "external", content: Layer["content"] = "unknown") {
    layer.version++
    layer.content = content
    if (origin === "user") layer.sync.touch()
    this.emit({ type: "layer", key: layer.key, origin })
  }

  private colorOf(color: PaintColor): Rgb {
    return color ?? this.tint
  }

  /** Sets who receives committed layer uploads and upload errors (the editor showing the document). */
  bindHandlers(handlers: { onCommit: (key: string, id: string | null) => void; onSyncError: (error: unknown) => void }) {
    this.onCommit = handlers.onCommit
    this.onSyncError = handlers.onSyncError
  }

  /** Whether a layer may have painted pixels (false: known empty). Cheap, for the UI. */
  touched(key: string): boolean {
    const content = this.layers.get(key)?.content
    return content !== undefined && content !== "empty"
  }

  /** Changes the display tint of mask layers (their pixels are recoloured; exports are unaffected). */
  setTint(tint: Rgb) {
    this.tint = tint
    for (const layer of this.layers.values()) {
      if (layer.kind !== "mask") continue
      layer.raster.tint(tint)
      this.emit({ type: "layer", key: layer.key, origin: "external" })
    }
  }

  /**
   * Replaces a layer's content from an image (its form value changed from
   * outside): masks read alpha, or luminance when the image is opaque.
   * The layer's history is dropped; `id` is what its value now is.
   */
  loadLayer(key: string, kind: LayerKind, source: CanvasImageSource | null, id: string | null) {
    const layer = this.ensureLayer(key, kind)
    if (this.stroke?.key === key) this.cancelStroke()
    layer.raster.clear()
    if (source) {
      layer.raster.drawImage(source)
      if (kind === "mask") {
        const full = this.fullRect()
        layer.raster.write(full, tintAlpha(maskAlphaFromImage(layer.raster.read(full)), this.tint))
      }
    }
    this.history.removeWhere((entry) => entry.layer === key)
    layer.sync.adopt(id)
    layer.bounds = source ? null : { rect: EMPTY_RECT, exact: true }
    this.changed(layer, "external", source ? "unknown" : "empty")
    this.emit({ type: "history" })
  }

  /** Empties a layer whose value was cleared from outside (not undoable: the value is gone). */
  resetLayer(key: string) {
    const layer = this.layers.get(key)
    if (!layer) return
    if (this.stroke?.key === key) this.cancelStroke()
    layer.raster.clear()
    this.history.removeWhere((entry) => entry.layer === key)
    layer.sync.adopt(null)
    layer.bounds = { rect: EMPTY_RECT, exact: true }
    this.changed(layer, "external", "empty")
    this.emit({ type: "history" })
  }

  private fullRect(): Rect {
    return { x: 0, y: 0, width: this.size.width, height: this.size.height }
  }

  /** The layer in wire format (mask: white + alpha), null when empty or missing. */
  exportLayer(key: string): LayerExport {
    const layer = this.layers.get(key)
    if (!layer || !this.hasImage) return null
    const data = layer.raster.read(this.fullRect())
    const any = layer.kind === "mask" ? toMaskExport(data) : hasContent(data)
    return any ? { data, width: this.size.width, height: this.size.height } : null
  }

  /**
   * The layer as a PNG in wire format, null when empty. The pixels are read
   * only when emptiness is unknown (after erasing, undo, a load); otherwise
   * the canvas is encoded straight away (masks: white with the mask's alpha,
   * composited on the GPU), so a large layer exports without a long task.
   */
  async exportPng(key: string): Promise<Blob | null> {
    const layer = this.layers.get(key)
    if (!layer || !this.hasImage || layer.content === "empty") return null
    if (layer.content === "unknown") {
      const empty = this.isEmpty(key)
      layer.content = empty ? "empty" : "painted"
      if (empty) return null
    }
    const png = layer.raster.toPng?.(layer.kind === "mask")
    if (png) return png
    const exported = this.exportLayer(key)
    return exported ? this.deps.sync.encode(exported.data, exported.width, exported.height) : null
  }

  /**
   * Bounds of the painted pixels of a layer. Tracked by the edits; pixels
   * are read only after an erase, undo or load, and then only inside the
   * known upper bound.
   */
  contentBounds(key: string): Rect {
    const layer = this.layers.get(key)
    if (!layer) return EMPTY_RECT
    const known = layer.bounds
    if (known?.exact) return known.rect
    const region = known ? clipRect(known.rect, this.size) : this.fullRect()
    let rect = EMPTY_RECT
    if (!isEmptyRect(region)) {
      const local = alphaBounds(layer.raster.read(region), region.width, region.height)
      if (!isEmptyRect(local)) rect = { x: local.x + region.x, y: local.y + region.y, width: local.width, height: local.height }
    }
    layer.bounds = { rect, exact: true }
    return rect
  }

  /** Records that an edit may have changed pixels in `rect` (`added`: it only added paint). */
  private touchBounds(layer: Layer, rect: Rect, added: boolean) {
    const known = layer.bounds
    if (!known) return
    layer.bounds = { rect: unionRect(known.rect, rect), exact: known.exact && added }
  }

  /** Whether a layer has any painted pixel. */
  isEmpty(key: string): boolean {
    return isEmptyRect(this.contentBounds(key))
  }

  // ------------------------------------------------------------ strokes

  get scratchRaster(): Raster | null {
    return this.scratch
  }

  private ensureScratch(): Raster {
    this.scratch ??= this.deps.createRaster(this.size.width, this.size.height)
    return this.scratch
  }

  /** The stroke in progress, drawn by the renderer over its layer. */
  get preview(): Preview | null {
    const s = this.stroke
    return s ? { key: s.key, rect: clipRect(s.rect, this.size), opacity: s.options.opacity, erase: s.options.erase } : null
  }

  get stroking() {
    return this.stroke !== null
  }

  beginStroke(key: string, kind: LayerKind, options: StrokeOptions) {
    this.cancelStroke()
    this.ensureLayer(key, kind)
    this.ensureScratch()
    const spec: DabSpec = { size: options.size, spacing: options.spacing ?? (options.hardness >= 0.98 ? 0.12 : 0.08), pressure: options.pressure }
    this.stroke = { key, walker: new StrokeWalker(spec), options, dabs: [], rect: EMPTY_RECT }
  }

  extendStroke(points: readonly StrokePoint[]) {
    const s = this.stroke
    if (!s || points.length === 0) return
    const dabs = s.walker.add(points)
    if (dabs.length === 0) return
    this.ensureScratch().dabs(dabs, s.options.hardness, this.colorOf(s.options.color))
    for (const value of dabs) s.dabs.push(value)
    const bounds = dabBounds(dabs, 2)
    if (bounds) s.rect = unionRect(s.rect, rectFromBounds(...bounds))
    this.emit({ type: "preview", key: s.key })
  }

  endStroke() {
    const s = this.stroke
    if (!s) return
    this.stroke = null
    const { options } = s
    this.commit(s.key, s.rect, {
      type: "paint",
      draw: { type: "dabs", dabs: Float32Array.from(s.dabs), hardness: options.hardness, color: options.color },
      opacity: options.opacity,
      erase: options.erase,
    })
  }

  cancelStroke() {
    const s = this.stroke
    if (!s) return
    this.stroke = null
    const rect = clipRect(s.rect, this.size)
    if (!isEmptyRect(rect)) this.scratch?.clear(rect)
    this.emit({ type: "preview", key: s.key })
  }

  // ------------------------------------------------------------ one-shot edits

  /** Fills a polygon `[x0, y0, …]` (rectangle and lasso tools). */
  polygon(key: string, kind: LayerKind, points: readonly number[], paint: PaintOptions) {
    if (points.length < 6) return
    this.ensureLayer(key, kind)
    const flat = Float32Array.from(points)
    const op: Op = { type: "paint", draw: { type: "polygon", points: flat, color: paint.color }, opacity: paint.opacity, erase: paint.erase }
    let x0 = Infinity
    let y0 = Infinity
    let x1 = -Infinity
    let y1 = -Infinity
    for (let i = 0; i + 1 < flat.length; i += 2) {
      x0 = Math.min(x0, flat[i]!)
      x1 = Math.max(x1, flat[i]!)
      y0 = Math.min(y0, flat[i + 1]!)
      y1 = Math.max(y1, flat[i + 1]!)
    }
    this.draw(op.draw)
    this.commit(key, rectFromBounds(x0 - 1, y0 - 1, x1 + 1, y1 + 1), op)
  }

  /**
   * Bucket fill at an image point: the region of similar colour in the layer
   * (`sample: "layer"`) or in the picture as shown (`"image"`: background and
   * the layer over it). Returns whether anything was filled.
   */
  fill(key: string, kind: LayerKind, x: number, y: number, tolerance: number, sample: "layer" | "image", paint: PaintOptions): boolean {
    if (!this.hasImage) return false
    const layer = this.ensureLayer(key, kind)
    const full = this.fullRect()
    const pixels = sample === "image" ? this.compositePixels(layer) : layer.raster.read(full)
    const region = floodFill(pixels, this.size.width, this.size.height, x, y, tolerance)
    if (region.count === 0) return false
    const { bounds } = region
    const alpha = new Uint8ClampedArray(bounds.width * bounds.height * 4)
    for (let row = 0; row < bounds.height; row++) {
      const src = (bounds.y + row) * this.size.width + bounds.x
      for (let col = 0; col < bounds.width; col++) if (region.mask[src + col]) alpha[(row * bounds.width + col) * 4 + 3] = 255
    }
    const draw: Draw = { type: "region", rect: bounds, alpha: encodePatch(alpha, bounds.width, bounds.height, "alpha"), color: paint.color }
    this.draw(draw)
    this.commit(key, bounds, { type: "paint", draw, opacity: paint.opacity, erase: paint.erase })
    return true
  }

  /** Inverts a mask layer (masked ↔ kept). */
  invert(key: string) {
    const layer = this.ensureLayer(key, "mask")
    const wasEmpty = layer.content === "empty"
    layer.raster.invert(this.tint)
    layer.bounds = null
    // Inverting twice gives the pixels back: nothing to keep for undo.
    this.record({ layer: key, rect: this.fullRect(), before: null, op: { type: "invert" }, bytes: 64 })
    this.changed(layer, "user", wasEmpty ? "painted" : "unknown")
  }

  /** Clears a layer (undoable). */
  clear(key: string) {
    const layer = this.layers.get(key)
    if (!layer || this.isEmpty(key)) return
    const rect = this.contentBounds(key)
    const before = this.snapshot(layer, tilesInRect(rect, this.size))
    layer.raster.clear()
    layer.bounds = { rect: EMPTY_RECT, exact: true }
    this.record({ layer: key, rect, before, op: { type: "clear" }, bytes: snapshotBytes(before) })
    this.changed(layer, "user", "empty")
  }

  // ------------------------------------------------------------ history

  /*
   * Undo and redo work per layer (`key`; every layer when omitted): the
   * editor undoes only the layer the user sees and edits, never a hidden
   * one of another mode. Edits of different layers commute, so taking the
   * newest entry of one layer out of the shared stack is exact.
   */

  /** Whether an edit of layer `key` (any layer when omitted) can be undone. */
  canUndo(key?: string | null): boolean {
    return !this.stroke && this.history.canUndoIn(scopeOf(key))
  }

  canRedo(key?: string | null): boolean {
    return !this.stroke && this.history.canRedoIn(scopeOf(key))
  }

  get historyDepth() {
    return this.history.depth
  }

  undo(key?: string | null): boolean {
    if (this.stroke) return false
    const entry = this.history.undo(scopeOf(key))
    if (!entry) return false
    const layer = this.layers.get(entry.layer)
    if (layer) {
      if (entry.before) for (const { rect, patch } of entry.before) layer.raster.write(rect, decodePatch(patch, this.tint))
      else this.apply(layer, entry.rect, entry.op)
      this.touchBounds(layer, entry.rect, false)
      this.changed(layer, "user")
    }
    this.emit({ type: "history" })
    return true
  }

  redo(key?: string | null): boolean {
    if (this.stroke) return false
    const entry = this.history.redo(scopeOf(key))
    if (!entry) return false
    const layer = this.layers.get(entry.layer)
    if (layer) {
      this.apply(layer, entry.rect, entry.op)
      if (entry.op.type === "clear") layer.bounds = { rect: EMPTY_RECT, exact: true }
      else this.touchBounds(layer, entry.rect, entry.op.type === "paint" && !entry.op.erase)
      this.changed(layer, "user")
    }
    this.emit({ type: "history" })
    return true
  }

  // ------------------------------------------------------------ sampling

  /** The colour shown at an image point (background, then the given layer over it). */
  sample(x: number, y: number, layerKey?: string | null): Rgb | null {
    if (!this.hasImage) return null
    const px = Math.floor(x)
    const py = Math.floor(y)
    if (px < 0 || py < 0 || px >= this.size.width || py >= this.size.height) return null
    const at: Rect = { x: px, y: py, width: 1, height: 1 }
    const base = this.backgroundRaster()?.read(at) ?? new Uint8ClampedArray([255, 255, 255, 255])
    const layer = layerKey ? this.layers.get(layerKey) : null
    const top = layer && layer.kind === "sketch" ? layer.raster.read(at) : null
    if (!top || top[3] === 0) return [base[0]!, base[1]!, base[2]!]
    const a = top[3]! / 255
    const mix = (c: number) => Math.round(top[c]! * a + base[c]! * (1 - a))
    return [mix(0), mix(1), mix(2)]
  }

  private backgroundRaster(): Raster | null {
    if (this.sampleRaster || !this.image || !this.hasImage) return this.sampleRaster
    const raster = this.deps.createRaster(this.size.width, this.size.height, { readFrequently: true })
    raster.drawImage(this.image)
    this.sampleRaster = raster
    return raster
  }

  /** Background with a sketch layer over it, as RGBA (fill sampling). */
  private compositePixels(layer: Layer): Uint8ClampedArray {
    const full = this.fullRect()
    const base = this.backgroundRaster()?.read(full) ?? new Uint8ClampedArray(full.width * full.height * 4).fill(255)
    if (layer.kind !== "sketch") return base
    const top = layer.raster.read(full)
    for (let i = 0; i < base.length; i += 4) {
      const a = top[i + 3]! / 255
      if (a === 0) continue
      for (let c = 0; c < 3; c++) base[i + c] = Math.round(top[i + c]! * a + base[i + c]! * (1 - a))
      base[i + 3] = 255
    }
    return base
  }

  // ------------------------------------------------------------ internals

  private draw(draw: Draw) {
    const scratch = this.ensureScratch()
    const color = this.colorOf(draw.color)
    if (draw.type === "dabs") scratch.dabs(draw.dabs, draw.hardness, color)
    else if (draw.type === "polygon") scratch.polygon(draw.points, color)
    else scratch.write(draw.rect, decodePatch(draw.alpha, color))
  }

  /** Composites the scratch's `rect` onto the layer and records the edit. */
  private commit(key: string, rect: Rect, op: Op & { type: "paint" }) {
    const layer = this.layers.get(key)
    const clipped = clipRect(rect, this.size)
    if (!layer || isEmptyRect(clipped)) {
      if (!isEmptyRect(clipped)) this.scratch?.clear(clipped)
      this.emit({ type: "preview", key })
      return
    }
    // Strokes keep only the tiles their dabs touch; shapes and fills their whole box.
    const tiles = op.draw.type === "dabs" ? tilesForDabs(op.draw.dabs, clipped) : tilesInRect(clipped, this.size)
    const before = this.snapshot(layer, tiles)
    layer.raster.composite(this.ensureScratch(), clipped, op.opacity, op.erase)
    this.scratch!.clear(clipped)
    this.touchBounds(layer, clipped, !op.erase)
    const drawBytes = op.draw.type === "dabs" ? op.draw.dabs.byteLength : op.draw.type === "polygon" ? op.draw.points.byteLength : patchBytes(op.draw.alpha)
    this.record({ layer: key, rect: clipped, before, op, bytes: snapshotBytes(before) + drawBytes })
    this.changed(layer, "user", op.erase ? "unknown" : "painted")
  }

  private snapshot(layer: Layer, tiles: readonly Rect[]): Snapshot {
    const format = formatOf(layer.kind)
    return tiles.map((rect) => ({ rect, patch: encodePatch(layer.raster.read(rect), rect.width, rect.height, format) }))
  }

  private record(entry: Entry) {
    // A new edit drops the redo branch of its own layer only.
    this.history.push(entry, (other) => other.layer === entry.layer)
    this.emit({ type: "history" })
  }

  /** Replays an edit (redo). */
  private apply(layer: Layer, rect: Rect, op: Op) {
    if (op.type === "invert") {
      layer.raster.invert(this.tint)
      layer.bounds = null
    } else if (op.type === "clear") layer.raster.clear()
    else {
      this.draw(op.draw)
      layer.raster.composite(this.ensureScratch(), rect, op.opacity, op.erase)
      this.scratch!.clear(rect)
    }
  }

  dispose() {
    this.cancelStroke()
    for (const layer of this.layers.values()) {
      layer.sync.dispose()
      disposeRaster(layer.raster)
    }
    this.layers.clear()
    if (this.scratch) disposeRaster(this.scratch)
    if (this.sampleRaster) disposeRaster(this.sampleRaster)
    this.scratch = this.sampleRaster = null
    this.history.clear()
    this.listeners.clear()
  }

  /** Flushes every layer's pending upload. */
  async flush(): Promise<void> {
    await Promise.all([...this.layers.values()].map((layer) => layer.sync.flush()))
  }

  get pending(): boolean {
    return [...this.layers.values()].some((layer) => layer.sync.pending)
  }
}

const scopeOf = (key: string | null | undefined) => (key ? (entry: Entry) => entry.layer === key : undefined)

function snapshotBytes(snapshot: Snapshot): number {
  return snapshot.reduce((sum, tile) => sum + patchBytes(tile.patch), 0)
}

function disposeRaster(raster: Raster) {
  ;(raster as { dispose?: () => void }).dispose?.()
}
