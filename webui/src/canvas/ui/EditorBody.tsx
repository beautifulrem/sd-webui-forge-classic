/**
 * One view of an editor document: the Konva surface with the brush cursor
 * and the checkerboard behind the image, the tool rail, the tool options,
 * the view bar, keyboard shortcuts, paste and drop. The inline editor and
 * the maximised one are two bodies over the same document.
 */
import {
  Brush,
  Eraser,
  Hand,
  ImageUp,
  Lasso,
  Loader2,
  Maximize2,
  Minimize2,
  Minus,
  PaintBucket,
  Pipette,
  Plus,
  Proportions,
  Redo2,
  Scan,
  SquareDashed,
  Undo2,
  X,
} from "@/components/icons"
import { useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { createStore, useStore } from "zustand"
import { Button } from "@/components/ui/button"
import { useAppearance } from "@/lib/theme"
import { cn } from "@/lib/utils"
import { usePasteTarget } from "@/ui/paste-targets"
import { fitView, isEmptyRect, nextZoomStep, type Rect, type View, zoomAt } from "../core/geometry"
import { aspectFrame, inpaintArea, RESIZE_MODES, scaledSize } from "../core/overlays"
import { parseHex, toHex } from "../core/pixels"
import { type Command, commandFor, isHovered, isMac, isTyping } from "../core/shortcuts"
import type { EditorDocument } from "../engine/document"
import { Interaction, type InteractionContext } from "../engine/interaction"
import { type Overlays, Renderer } from "../engine/renderer"
import { type BrushSettings, stepSize, type Tool, useTools } from "../state/tools"
import type { ImageEditorProps } from "../types"
import { toolbarKeys } from "./a11y"
import { Divider, Hint, IconButton, PANEL } from "./controls"
import { EmptyState } from "./EmptyState"
import { LayersPopover } from "./LayersPopover"
import { OptionsBar } from "./OptionsBar"
import { ShortcutsPopover } from "./ShortcutsPopover"
import { useDocValue, useLayerPending } from "./use-doc"

type BodyProps = {
  doc: EditorDocument
  editor: ImageEditorProps
  /** Brush sizes in screen px (`forge_canvas_consistent_brush`). */
  screenBrush: boolean
  /** Solid backdrop behind transparent pixels instead of the checkerboard (`forge_canvas_plain_color`). */
  plainColor: string | null
  /** Toolbars always shown (`forge_canvas_toolbar_always`); otherwise while the pointer is over the editor. */
  toolbarAlways: boolean
  maximized: boolean
  onMaximize: (maximized: boolean) => void
  /** The background is uploading or loading. */
  busy: boolean
  /** The document shows the form's image (painting waits for a new one to load). */
  ready: boolean
  /** The form's image could not be loaded: an error with Retry instead of the editor. */
  loadFailed: boolean
  onRetry: () => void
  onFiles: (files: File[]) => void
  onBlank: () => void
  onRemove: () => void
}

/** How long the resize frame / inpaint previews stay after their setting changed. */
const FLASH_MS = 2200

const TOOLS: { tool: Tool; label: string; key: string; icon: typeof Brush }[] = [
  { tool: "brush", label: "Brush", key: "B", icon: Brush },
  { tool: "eraser", label: "Eraser", key: "E", icon: Eraser },
  { tool: "fill", label: "Fill", key: "G", icon: PaintBucket },
  { tool: "rect", label: "Rectangle", key: "M", icon: SquareDashed },
  { tool: "lasso", label: "Lasso", key: "L", icon: Lasso },
  { tool: "eyedropper", label: "Eyedropper", key: "I", icon: Pipette },
  { tool: "hand", label: "Hand", key: "H", icon: Hand },
]

const CURSORS = { brush: "crosshair", eraser: "crosshair", cross: "crosshair", pick: "crosshair", grab: "grab", grabbing: "grabbing", default: "default" } as const

/** A CSS colour (e.g. the theme's oklch primary) as `#rrggbb` for the canvas overlays. */
function cssToHex(color: string): string | null {
  if (!color) return null
  const canvas = document.createElement("canvas")
  canvas.width = canvas.height = 1
  const ctx = canvas.getContext("2d")
  if (!ctx) return null
  ctx.fillStyle = "#000"
  ctx.fillStyle = color
  ctx.fillRect(0, 0, 1, 1)
  const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data
  return toHex([r!, g!, b!])
}

/**
 * True for `FLASH_MS` after `key` changes from one value to another (not on
 * mount, nor when the setting appears or goes: `null`, e.g. a mode switch).
 */
function useFlash(key: string | null): boolean {
  const [last, setLast] = useState(key)
  const [flashing, setFlashing] = useState<string | null>(null)
  if (last !== key) {
    setLast(key)
    if (last !== null && key !== null) setFlashing(key)
  }
  useEffect(() => {
    if (flashing === null) return
    const timer = setTimeout(() => setFlashing(null), FLASH_MS)
    return () => clearTimeout(timer)
  }, [flashing])
  return flashing === key
}

/** The brush settings an embedding widget pins (unset ones stay the user's). */
function withFixed(fixed: { size?: number; opacity?: number; softness?: number }): Partial<BrushSettings> {
  const out: Partial<BrushSettings> = {}
  if (fixed.size !== undefined) out.size = fixed.size
  if (fixed.opacity !== undefined) out.opacity = fixed.opacity
  if (fixed.softness !== undefined) out.softness = fixed.softness
  return out
}

/** View operations of a mounted body (created with its renderer). */
type ViewOps = {
  fit: () => void
  /** Fits again unless the user zoomed or panned since the last fit. */
  refit: () => void
  zoomTo: (next: (view: View) => View) => void
  center: () => { x: number; y: number }
}

/** Whether a tool is offered: the eyedropper only on a sketch layer whose colour the user picks. */
const offered = (tool: Tool, pickable: boolean) => tool !== "eyedropper" || pickable

export function EditorBody({ doc, editor, screenBrush, plainColor, toolbarAlways, maximized, onMaximize, busy, ready, loadFailed, onRetry, onFiles, onBlank, onRemove }: BodyProps) {
  const { t } = useTranslation()
  const mod = isMac() ? "⌘" : "Ctrl"
  const rootRef = useRef<HTMLDivElement>(null)
  const hostRef = useRef<HTMLDivElement>(null)
  const backdropRef = useRef<HTMLDivElement>(null)
  const cursorRef = useRef<HTMLDivElement>(null)
  const rendererRef = useRef<Renderer | null>(null)
  const interactionRef = useRef<Interaction | null>(null)
  const opsRef = useRef<ViewOps | null>(null)
  const viewRef = useRef<View>({ scale: 1, x: 0, y: 0 })
  const overlaysRef = useRef<Overlays>({})
  const hovered = useRef(false)
  const adjustKey = useRef<InteractionContext["adjust"]>(null)
  const toolBeforePick = useRef<Tool | null>(null)
  const [viewStore] = useState(() => createStore<{ scale: number }>(() => ({ scale: 1 })))
  const scale = useStore(viewStore, (s) => s.scale)
  const [dragging, setDragging] = useState(false)
  const [helpOpen, setHelpOpen] = useState(false)
  const [area, setArea] = useState<Rect | null>(null)

  const tools = useTools()
  const accent = useAppearance((s) => s.accent)
  const hasImage = useDocValue(doc, (d) => d.hasImage)
  const width = useDocValue(doc, (d) => d.width)
  const height = useDocValue(doc, (d) => d.height)
  const activeKey = editor.noScribble ? null : (editor.activeLayer ?? null)
  // Undo and redo follow the layer on screen, never a hidden one of another mode.
  const canUndo = useDocValue(doc, (d) => Boolean(activeKey) && d.canUndo(activeKey))
  const canRedo = useDocValue(doc, (d) => Boolean(activeKey) && d.canRedo(activeKey))
  const active = activeKey ? (editor.layers?.[activeKey] ?? null) : null
  const kind = active?.kind ?? null
  const touched = useDocValue(doc, (d) => (activeKey ? d.touched(activeKey) : false))
  const pending = useLayerPending(doc, activeKey)
  const paintable = Boolean(active && hasImage && ready)
  const pickable = kind === "sketch" && !editor.fixedColor
  // A tool this layer does not offer (the eyedropper on a mask) falls back to the brush.
  const tool: Tool = paintable ? (offered(tools.tool, pickable) ? tools.tool : "brush") : "hand"
  const layerLabel = active ? (active.label ? t(active.label) : kind === "mask" ? t("Mask") : t("Sketch")) : null
  const fixed = { size: editor.fixedSize, opacity: editor.fixedOpacity, softness: editor.fixedSoftness }

  // What the event handlers read (they live as long as the renderer).
  const snapshot = () => ({ activeKey, kind, pickable, fixedColor: editor.fixedColor, fixed, screenBrush, paintable, noUpload: editor.noUpload, onFiles })
  const latest = useRef(snapshot())
  useEffect(() => {
    latest.current = snapshot()
  })
  // A paste with no image field hovered or focused goes to the first empty one on screen (this editor while it has no image).
  usePasteTarget(!editor.noUpload, { element: () => rootRef.current, empty: () => !hasImage, deliver: (files) => onFiles(files) })

  // ------------------------------------------------------------ renderer, input, view

  useEffect(() => {
    const host = hostRef.current
    const cursor = cursorRef.current
    if (!host || !cursor || !hasImage) return
    const renderer = new Renderer(doc)
    renderer.element.style.zIndex = "1"
    renderer.attach(host)
    rendererRef.current = renderer
    let autoFit = true

    const viewport = () => ({ width: host.clientWidth, height: host.clientHeight })
    const applyView = (view: View) => {
      viewRef.current = view
      renderer.setView(view)
      const backdrop = backdropRef.current
      if (backdrop) {
        backdrop.style.transform = `translate(${view.x}px, ${view.y}px)`
        backdrop.style.width = `${doc.width * view.scale}px`
        backdrop.style.height = `${doc.height * view.scale}px`
      }
      viewStore.setState({ scale: view.scale })
    }
    const fit = () => {
      // Room for the floating toolbars around the image.
      const pad = latest.current.paintable ? { left: 56, top: 56, right: 12, bottom: 56 } : { left: 12, top: 12, right: 12, bottom: 56 }
      const box = viewport()
      const view = fitView({ width: doc.width, height: doc.height }, { width: box.width - pad.left - pad.right, height: box.height - pad.top - pad.bottom }, { margin: 0, maxScale: 8 })
      autoFit = true
      applyView({ scale: view.scale, x: view.x + pad.left, y: view.y + pad.top })
    }
    opsRef.current = {
      fit,
      refit: () => {
        if (autoFit) fit()
      },
      zoomTo: (next) => {
        autoFit = false
        applyView(next(viewRef.current))
      },
      center: () => ({ x: host.clientWidth / 2, y: host.clientHeight / 2 }),
    }

    const interaction = new Interaction(host, doc, {
      context: () => {
        const s = useTools.getState()
        const l = latest.current
        const layer = l.paintable && l.activeKey && l.kind ? { key: l.activeKey, kind: l.kind } : null
        const pinned = withFixed(l.fixed)
        return {
          layer,
          tool: layer ? (offered(s.tool, l.pickable) ? s.tool : "brush") : "hand",
          brush: { ...s.brush, ...pinned },
          eraser: { ...s.eraser, ...pinned },
          color: layer?.kind === "mask" ? null : (parseHex(l.fixedColor ?? s.color) ?? [0, 0, 0]),
          fillTolerance: s.fillTolerance,
          fillSample: s.fillSample,
          pressure: s.pressure,
          screenBrush: l.screenBrush,
          wheel: s.wheel,
          adjust: adjustKey.current,
        }
      },
      view: () => viewRef.current,
      viewport,
      setView: (view) => {
        autoFit = false
        applyView(view)
      },
      picked: (rgb, final) => {
        const s = useTools.getState()
        s.setColor(toHex(rgb))
        if (!final) return
        const back = toolBeforePick.current
        s.setTool(back && back !== "eyedropper" ? back : "brush")
        toolBeforePick.current = null
      },
      cursor: (state) => {
        if (!state) {
          cursor.style.display = "none"
          host.style.cursor = ""
          return
        }
        const ring = (state.mode === "brush" || state.mode === "eraser") && state.diameter >= 3
        cursor.style.display = ring ? "block" : "none"
        if (ring) {
          const d = state.diameter
          cursor.style.width = cursor.style.height = `${d}px`
          cursor.style.transform = `translate(${state.x - d / 2}px, ${state.y - d / 2}px)`
          cursor.dataset.mode = state.mode
        }
        host.style.cursor = ring ? "none" : CURSORS[state.mode]
      },
      shape: (points, erase) => {
        overlaysRef.current = { ...overlaysRef.current, shape: points, shapeErase: erase }
        renderer.setOverlays(overlaysRef.current)
      },
      adjust: (what, direction) => {
        const s = useTools.getState()
        const eraser = s.tool === "eraser"
        const brush = eraser ? s.eraser : s.brush
        const set = eraser ? s.setEraser : s.setBrush
        if (what === "size") set({ size: stepSize(brush.size, direction) })
        else if (what === "opacity") set({ opacity: brush.opacity + direction * 0.05 })
        else set({ softness: brush.softness + direction * 0.05 })
      },
      painted: (color) => {
        if (color) useTools.getState().rememberColor(toHex(color))
      },
    })
    interactionRef.current = interaction

    let sized = false
    const observer = new ResizeObserver(() => {
      const box = viewport()
      if (box.width === 0 || box.height === 0) return
      const before = { width: renderer.element.clientWidth, height: renderer.element.clientHeight }
      renderer.setSize(box.width, box.height)
      if (!sized || autoFit) {
        sized = true
        fit()
        return
      }
      // Keep what was in the middle in the middle (both ways: the stage also changes height).
      const view = viewRef.current
      applyView({ ...view, x: view.x + (box.width - before.width) / 2, y: view.y + (box.height - before.height) / 2 })
    })
    observer.observe(host)
    const unsubscribe = doc.subscribe((event) => {
      if (event.type === "background" && doc.hasImage) fit()
    })
    return () => {
      unsubscribe()
      observer.disconnect()
      interaction.destroy()
      renderer.destroy()
      rendererRef.current = null
      interactionRef.current = null
      opsRef.current = null
    }
  }, [doc, hasImage, viewStore])

  // ------------------------------------------------------------ layer, style, tint, accent

  useEffect(() => {
    rendererRef.current?.setLayer(activeKey)
  }, [activeKey, hasImage])

  // The tool rail and options take room around the image: fit again when they come or go.
  useEffect(() => {
    opsRef.current?.refit()
  }, [paintable])

  useEffect(() => {
    const tint = parseHex(tools.maskColor)
    if (tint) doc.setTint(tint)
  }, [doc, tools.maskColor])

  useEffect(() => {
    const color = cssToHex(getComputedStyle(rootRef.current ?? document.documentElement).getPropertyValue("--primary").trim())
    if (color) rendererRef.current?.setAccent(color)
  }, [accent, hasImage])

  // Inpaint previews: on while their setting is being changed, or kept on by the user.
  const inpaint = editor.inpaint ?? null
  const scaleBy = editor.frame?.scale ?? null
  // "Resize by": the processing size follows the image.
  const scaled = scaleBy && width > 0 && height > 0 ? scaledSize({ width, height }, scaleBy.by, scaleBy.step) : null
  const targetWidth = scaled?.width ?? editor.frame?.width ?? 0
  const targetHeight = scaled?.height ?? editor.frame?.height ?? 0
  const resizeMode = editor.frame?.resizeMode ?? null
  const blurFlash = useFlash(inpaint ? String(inpaint.blur) : null)
  const areaFlash = useFlash(inpaint ? `${inpaint.padding}|${inpaint.onlyMasked}|${targetWidth}x${targetHeight}` : null)
  const frameFlash = useFlash(resizeMode !== null ? `${targetWidth}x${targetHeight}|${resizeMode}|${scaleBy?.by ?? ""}` : null)
  const blur = kind === "mask" && inpaint && (tools.previewBlur || blurFlash) ? inpaint.blur : 0
  const showFrame = frameFlash && hasImage && targetWidth > 0 && targetHeight > 0 && resizeMode !== null
  const showArea = Boolean(inpaint?.onlyMasked && kind && hasImage && targetWidth > 0 && (tools.previewArea || areaFlash))

  useEffect(() => {
    const opacity = kind === "mask" ? tools.maskOpacity : tools.sketchOpacity
    rendererRef.current?.setStyle({ visible: tools.layerVisible, opacity, imageVisible: tools.imageVisible, blur })
  }, [kind, tools.layerVisible, tools.maskOpacity, tools.sketchOpacity, tools.imageVisible, blur, hasImage])

  // The "Only masked" area follows the mask while it is shown.
  const blurValue = inpaint?.blur ?? 0
  const padding = inpaint?.padding ?? 0
  const inverted = Boolean(inpaint?.inverted)
  useEffect(() => {
    if (!showArea || !activeKey) return
    const compute = () =>
      setArea(
        inpaintArea({
          maskBounds: doc.contentBounds(activeKey),
          image: { width: doc.width, height: doc.height },
          target: { width: targetWidth, height: targetHeight },
          blur: blurValue,
          padding,
          inverted,
        }),
      )
    let timer = setTimeout(compute, 0)
    const unsubscribe = doc.subscribe((event) => {
      if (event.type !== "layer" || event.key !== activeKey) return
      clearTimeout(timer)
      timer = setTimeout(compute, 120)
    })
    return () => {
      clearTimeout(timer)
      unsubscribe()
    }
  }, [doc, showArea, activeKey, targetWidth, targetHeight, blurValue, padding, inverted])

  const frame = showFrame && resizeMode !== null ? aspectFrame({ width, height }, { width: targetWidth, height: targetHeight }, resizeMode) : null
  const shownArea = showArea && area && !isEmptyRect(area) ? area : null
  useEffect(() => {
    const next = showFrame && resizeMode !== null ? aspectFrame({ width, height }, { width: targetWidth, height: targetHeight }, resizeMode) : null
    overlaysRef.current = { ...overlaysRef.current, frame: next, area: shownArea }
    rendererRef.current?.setOverlays(overlaysRef.current)
  }, [showFrame, width, height, targetWidth, targetHeight, resizeMode, shownArea, hasImage])

  // ------------------------------------------------------------ commands and keys

  const setTool = (next: Tool) => {
    if (next === "eyedropper" && tool !== "eyedropper") toolBeforePick.current = tool
    tools.setTool(next)
  }

  const run = (command: Command) => {
    const ops = opsRef.current
    switch (command.type) {
      case "tool":
        if (paintable && offered(command.tool, pickable)) setTool(command.tool)
        return
      case "undo":
        if (paintable && !interactionRef.current?.busy) doc.undo(activeKey)
        return
      case "redo":
        if (paintable && !interactionRef.current?.busy) doc.redo(activeKey)
        return
      case "size": {
        if (editor.fixedSize !== undefined) return
        const eraser = tool === "eraser"
        const brush = eraser ? tools.eraser : tools.brush
        ;(eraser ? tools.setEraser : tools.setBrush)({ size: stepSize(brush.size, command.direction) })
        return
      }
      case "clear":
        if (activeKey && paintable) doc.clear(activeKey)
        return
      case "invert":
        if (activeKey && kind === "mask" && paintable) doc.invert(activeKey)
        return
      case "fit":
        ops?.fit()
        return
      case "actual":
        ops?.zoomTo((view) => zoomAt(view, 1 / view.scale, ops.center()))
        return
      case "zoom":
        ops?.zoomTo((view) => zoomAt(view, nextZoomStep(view.scale, command.direction) / view.scale, ops.center()))
        return
      case "maximize":
        onMaximize(!maximized)
        return
      case "help":
        setHelpOpen(true)
        return
    }
  }
  const runRef = useRef(run)
  useEffect(() => {
    runRef.current = run
  })

  useEffect(() => {
    // The editor takes the keys while the pointer is over it or focus is in it. Pointer
    // enter alone misses an editor that mounts under a resting pointer (e.g. after "Send
    // to"), so any pointer move or press inside counts too, and so does :hover.
    const isActive = () => {
      const root = rootRef.current
      return Boolean(root && (hovered.current || isHovered(root) || root.contains(document.activeElement)))
    }
    const reset = () => {
      adjustKey.current = null
      interactionRef.current?.setSpace(false)
      interactionRef.current?.setShift(false)
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || !isActive() || isTyping(event.target)) return
      if (event.code === "Space") {
        event.preventDefault()
        interactionRef.current?.setSpace(true)
        return
      }
      if (event.key === "Shift") interactionRef.current?.setShift(true)
      const plain = !event.ctrlKey && !event.metaKey && !event.altKey
      if (plain && (event.code === "KeyW" || event.code === "KeyA" || event.code === "KeyS")) {
        adjustKey.current = event.code === "KeyW" ? "size" : event.code === "KeyA" ? "opacity" : "softness"
        return
      }
      if (event.repeat && event.key !== "[" && event.key !== "]") return
      const command = commandFor(event, isMac())
      if (!command) return
      event.preventDefault()
      runRef.current(command)
    }
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.code === "Space") interactionRef.current?.setSpace(false)
      if (event.key === "Shift") interactionRef.current?.setShift(false)
      if (event.code === "KeyW" || event.code === "KeyA" || event.code === "KeyS") adjustKey.current = null
    }
    const onPaste = (event: ClipboardEvent) => {
      if (latest.current.noUpload || !isActive() || isTyping(event.target)) return
      const files = Array.from(event.clipboardData?.files ?? []).filter((file) => file.type.startsWith("image/"))
      if (files.length === 0) return
      event.preventDefault()
      latest.current.onFiles(files)
    }
    window.addEventListener("keydown", onKeyDown)
    window.addEventListener("keyup", onKeyUp)
    window.addEventListener("blur", reset)
    window.addEventListener("paste", onPaste)
    return () => {
      window.removeEventListener("keydown", onKeyDown)
      window.removeEventListener("keyup", onKeyUp)
      window.removeEventListener("blur", reset)
      window.removeEventListener("paste", onPaste)
    }
  }, [])

  // ------------------------------------------------------------ render

  const modeName = typeof resizeMode === "number" ? (RESIZE_MODES[resizeMode] ?? "") : (resizeMode ?? "")
  const frameLabel = frame ? `${targetWidth}×${targetHeight} · ${t(modeName)}` : null
  const areaLabel = shownArea ? t("Inpaint area {{width}}×{{height}}", { width: shownArea.width, height: shownArea.height }) : null
  const editorName = t("Image editor, {{width}}×{{height}}", { width, height })

  return (
    <div
      ref={rootRef}
      data-slot="image-editor"
      className="group/editor @container relative isolate flex size-full min-h-0 flex-col overflow-hidden bg-canvas"
      onPointerEnter={() => (hovered.current = true)}
      onPointerMoveCapture={() => (hovered.current = true)}
      onPointerDownCapture={() => (hovered.current = true)}
      onPointerLeave={() => (hovered.current = false)}
      onDragOver={(event) => {
        if (editor.noUpload || !event.dataTransfer.types.includes("Files")) return
        event.preventDefault()
        event.dataTransfer.dropEffect = "copy"
        setDragging(true)
      }}
      onDragLeave={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false)
      }}
      onDrop={(event) => {
        if (editor.noUpload) return
        event.preventDefault()
        setDragging(false)
        const files = Array.from(event.dataTransfer.files).filter((file) => file.type.startsWith("image/"))
        if (files.length > 0) onFiles(files)
      }}
    >
      {hasImage ? (
        <div
          ref={hostRef}
          role="img"
          tabIndex={0}
          aria-label={layerLabel ? `${editorName} · ${layerLabel}` : editorName}
          // Nothing to paint: one finger scrolls the page (two fingers pan and zoom the image).
          className={cn(
            "absolute inset-0 overflow-hidden outline-none select-none focus-visible:ring-3 focus-visible:ring-ring/40 focus-visible:ring-inset",
            paintable ? "touch-none" : "touch-pan-y",
          )}
        >
          <div
            ref={backdropRef}
            aria-hidden
            className="pointer-events-none absolute top-0 left-0 z-0 origin-top-left bg-checker shadow-[0_0_0_1px_var(--border),0_8px_30px_-12px_rgb(0_0_0/0.35)]"
            style={plainColor ? { background: plainColor } : undefined}
          />
          <div
            ref={cursorRef}
            aria-hidden
            className="pointer-events-none absolute top-0 left-0 z-[2] hidden rounded-full border border-white shadow-[0_0_0_1px_rgb(0_0_0/0.55),inset_0_0_0_1px_rgb(0_0_0/0.55)] data-[mode=eraser]:border-dashed"
          />
        </div>
      ) : (
        <EmptyState
          busy={busy}
          noUpload={editor.noUpload}
          blankSize={editor.blankSize}
          hint={editor.hint}
          error={editor.error}
          loadFailed={loadFailed}
          onRetry={onRetry}
          onFiles={onFiles}
          onBlank={onBlank}
        />
      )}

      {busy && hasImage ? (
        <div className="pointer-events-none absolute inset-0 z-10 grid place-items-center bg-background/40">
          <Loader2 className="size-6 animate-spin text-muted-foreground" aria-hidden />
        </div>
      ) : null}

      {hasImage ? (
        <div
          className={cn(
            "pointer-events-none absolute inset-0 z-20",
            !toolbarAlways &&
              "opacity-0 transition-opacity duration-200 group-focus-within/editor:opacity-100 group-hover/editor:opacity-100 [@media(hover:none)]:opacity-100",
          )}
        >
          {paintable ? (
            <div
              role="toolbar"
              aria-label={t("Tools")}
              aria-orientation="vertical"
              onKeyDown={toolbarKeys("vertical")}
              className={cn(PANEL, "absolute top-2 left-2 flex max-h-[calc(100%-4.5rem)] flex-col items-center gap-0.5 overflow-y-auto p-1 [scrollbar-width:none]")}
            >
              {TOOLS.filter((item) => offered(item.tool, pickable)).map(({ tool: value, label, key, icon: Icon }) => (
                <IconButton key={value} label={t(label)} keys={[key]} pressed={tool === value} onClick={() => setTool(value)} side="right">
                  <Icon />
                </IconButton>
              ))}
            </div>
          ) : null}

          {paintable && kind ? (
            <div className="absolute top-2 right-2 left-14 flex justify-start @2xl:justify-center">
              <OptionsBar
                tool={tool}
                kind={kind}
                fixedColor={editor.fixedColor}
                fixedSize={editor.fixedSize}
                fixedOpacity={editor.fixedOpacity}
                fixedSoftness={editor.fixedSoftness}
                canClear={touched}
                mod={mod}
                onInvert={() => run({ type: "invert" })}
                onClear={() => run({ type: "clear" })}
              />
            </div>
          ) : null}

          {editor.error ? (
            <p role="alert" className={cn(PANEL, "absolute top-14 left-1/2 max-w-[calc(100%-7rem)] -translate-x-1/2 border-destructive/40 px-3 py-1.5 text-sm text-destructive")}>
              {t(editor.error)}
            </p>
          ) : null}

          {frameLabel || areaLabel ? (
            <div className="absolute top-14 right-2 flex flex-col items-end gap-1" aria-live="polite">
              {frameLabel ? <span className={cn(PANEL, "px-2 py-1 text-xs tabular-nums")}>{frameLabel}</span> : null}
              {areaLabel ? <span className={cn(PANEL, "px-2 py-1 text-xs text-primary tabular-nums")}>{areaLabel}</span> : null}
            </div>
          ) : null}

          <div className="absolute inset-x-2 bottom-2 flex items-end justify-between gap-2">
            <div role="toolbar" aria-label={t("History and zoom")} data-toast-obstacle onKeyDown={toolbarKeys("horizontal")} className={cn(PANEL, "flex h-10 items-center gap-0.5 px-1")}>
              {paintable ? (
                <>
                  <IconButton label={t("Undo")} keys={[mod, "Z"]} onClick={() => run({ type: "undo" })} disabled={!canUndo}>
                    <Undo2 />
                  </IconButton>
                  <IconButton label={t("Redo")} keys={[mod, "Y"]} onClick={() => run({ type: "redo" })} disabled={!canRedo}>
                    <Redo2 />
                  </IconButton>
                  <Divider />
                </>
              ) : null}
              <span className="hidden @md:contents">
                <IconButton label={t("Zoom out")} keys={["−"]} onClick={() => run({ type: "zoom", direction: -1 })}>
                  <Minus />
                </IconButton>
              </span>
              <Hint label={t("Actual size")} keys={["1"]}>
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-14 px-1 tabular-nums"
                  aria-label={t("Zoom {{percent}}%, show at actual size", { percent: Math.round(scale * 100) })}
                  onClick={() => run({ type: "actual" })}
                >
                  {Math.round(scale * 100)}%
                </Button>
              </Hint>
              <span className="hidden @md:contents">
                <IconButton label={t("Zoom in")} keys={["+"]} onClick={() => run({ type: "zoom", direction: 1 })}>
                  <Plus />
                </IconButton>
              </span>
              <IconButton label={t("Fit and center")} keys={["R"]} onClick={() => run({ type: "fit" })}>
                <Scan />
              </IconButton>
            </div>

            <div role="toolbar" aria-label={t("Image")} data-toast-obstacle onKeyDown={toolbarKeys("horizontal")} className={cn(PANEL, "flex h-10 min-w-0 items-center gap-0.5 px-1")}>
              {pending ? (
                <span role="status" className="flex items-center gap-1.5 px-1.5 text-xs text-muted-foreground">
                  <Loader2 className="size-3 animate-spin" aria-hidden />
                  <span className="hidden @lg:inline">{t("Saving…")}</span>
                </span>
              ) : null}
              <span className="hidden px-1.5 text-xs text-muted-foreground tabular-nums @xl:inline">
                {width}×{height}
              </span>
              {editor.onUseImageSize ? (
                <IconButton label={t("Use the image size ({{width}}×{{height}})", { width, height })} onClick={() => editor.onUseImageSize?.({ width, height })}>
                  <Proportions />
                </IconButton>
              ) : null}
              <LayersPopover layer={kind && layerLabel ? { kind, label: layerLabel } : null} inpaint={Boolean(inpaint)} onlyMasked={Boolean(inpaint?.onlyMasked)} />
              <span className="hidden @md:contents">
                <ShortcutsPopover open={helpOpen} onOpenChange={setHelpOpen} />
              </span>
              {editor.noUpload ? null : (
                <>
                  <Divider />
                  <ReplaceButton onFiles={onFiles} />
                  <IconButton label={t("Remove image")} onClick={onRemove}>
                    <X />
                  </IconButton>
                </>
              )}
              <Divider />
              <IconButton label={maximized ? t("Exit full screen") : t("Edit in full screen")} keys={["F"]} onClick={() => onMaximize(!maximized)}>
                {maximized ? <Minimize2 /> : <Maximize2 />}
              </IconButton>
            </div>
          </div>
        </div>
      ) : null}

      {dragging ? (
        <div className="pointer-events-none absolute inset-2 z-30 grid place-items-center rounded-md border-2 border-dashed border-primary bg-primary/10 text-sm font-medium text-primary">
          {hasImage ? t("Drop to replace the image") : t("Drop the image here")}
        </div>
      ) : null}
    </div>
  )
}

function ReplaceButton({ onFiles }: { onFiles: (files: File[]) => void }) {
  const { t } = useTranslation()
  const input = useRef<HTMLInputElement>(null)
  return (
    <>
      <IconButton label={t("Replace image")} onClick={() => input.current?.click()}>
        <ImageUp />
      </IconButton>
      <input
        ref={input}
        type="file"
        accept="image/*"
        hidden
        onChange={(event) => {
          const files = Array.from(event.target.files ?? [])
          event.target.value = ""
          if (files.length > 0) onFiles(files)
        }}
      />
    </>
  )
}
