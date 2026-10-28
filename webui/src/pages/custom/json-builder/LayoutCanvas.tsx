/**
 * The layout canvas: the draft's items as boxes on a stage of the draft's
 * aspect ratio. Drag a box to move it, its corner to resize it (arrow keys
 * move the focused box, Shift for 16 px); double-click its caption (or text)
 * to edit it in place. The wheel zooms at the cursor, the middle button (or
 * the left one on the background while zoomed in) pans, a double-click on the
 * background fits the stage again.
 */
import { Maximize, ZoomIn, ZoomOut } from "@/components/icons"
import { type CSSProperties, type KeyboardEvent, type PointerEvent as ReactPointerEvent, useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { type Item, LABEL_NAMES, type Missing, moveBox, resizeBox } from "./draft"
import { useBuilder } from "./store"
import { clampView, fitStage, RESET_VIEW, type View, ZOOM_STEP, zoomAbout } from "./view"

/** The stage's margin inside the canvas area, in CSS pixels (each side). */
const INSET = 16
/** A press moves a box once the pointer went this far (in canvas pixels). */
const DRAG_THRESHOLD = 3

type Size = { width: number; height: number }

function useElementSize(element: HTMLElement | null): Size {
  const [size, setSize] = useState<Size>({ width: 0, height: 0 })
  useEffect(() => {
    if (!element) return
    const update = () => setSize({ width: element.clientWidth, height: element.clientHeight })
    update()
    const observer = new ResizeObserver(update)
    observer.observe(element)
    return () => observer.disconnect()
  }, [element])
  return size
}

/** Follows the pointer until it is released, calling `move` with its offset from `start` in client pixels. */
function track(start: { clientX: number; clientY: number }, move: (dx: number, dy: number) => void, end?: () => void) {
  const onMove = (event: PointerEvent) => move(event.clientX - start.clientX, event.clientY - start.clientY)
  const onUp = () => {
    window.removeEventListener("pointermove", onMove)
    window.removeEventListener("pointerup", onUp)
    window.removeEventListener("pointercancel", onUp)
    end?.()
  }
  window.addEventListener("pointermove", onMove)
  window.addEventListener("pointerup", onUp)
  window.addEventListener("pointercancel", onUp)
}

type Field = "caption" | "text"

function BoxView({
  item,
  index,
  stage,
  active,
  missing,
  onActivate,
}: {
  item: Item
  /** Its position in the list (shown 1-based). */
  index: number
  stage: HTMLElement | null
  active: boolean
  missing: ReadonlySet<Missing>
  onActivate: (id: number | null) => void
}) {
  const { t } = useTranslation()
  const width = useBuilder((s) => s.width)
  const height = useBuilder((s) => s.height)
  const setBox = useBuilder((s) => s.setBox)
  const updateItem = useBuilder((s) => s.updateItem)
  const [editing, setEditing] = useState<Field | null>(null)
  const [dragging, setDragging] = useState(false)
  const [x1, y1, x2, y2] = item.bbox

  /** Canvas pixels per client pixel (the stage's on-screen size includes the zoom). */
  const pixelScale = () => {
    const rect = stage?.getBoundingClientRect()
    return rect && rect.width > 0 && rect.height > 0 ? { x: width / rect.width, y: height / rect.height } : null
  }

  const startDrag = (event: ReactPointerEvent) => {
    const scale = pixelScale()
    if (!scale) return
    const from = item.bbox
    let moved = false
    track(
      event,
      (dx, dy) => {
        const px = dx * scale.x
        const py = dy * scale.y
        if (!moved && Math.hypot(px, py) < DRAG_THRESHOLD) return
        moved = true
        setDragging(true)
        setBox(item.id, moveBox(from, px, py, width, height))
      },
      () => setDragging(false),
    )
  }

  const startResize = (event: ReactPointerEvent) => {
    const scale = pixelScale()
    if (!scale) return
    const from = item.bbox
    setDragging(true)
    track(
      event,
      (dx, dy) => setBox(item.id, resizeBox(from, dx * scale.x, dy * scale.y, width, height)),
      () => setDragging(false),
    )
  }

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.target !== event.currentTarget) return
    const step = event.shiftKey ? 16 : 1
    const delta = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[event.key]
    if (!delta) return
    event.preventDefault()
    setBox(item.id, moveBox(item.bbox, delta[0]!, delta[1]!, width, height))
  }

  const field = (name: Field) => {
    const invalid = missing.has(`item:${item.id}:${name}`)
    return (
      <textarea
        aria-label={name === "caption" ? t("Box {{n}} caption", { n: index + 1 }) : t("Box {{n}} text", { n: index + 1 })}
        aria-invalid={invalid || undefined}
        readOnly={editing !== name}
        spellCheck={false}
        placeholder={name === "caption" ? t("Caption") : t("Text to render")}
        value={name === "caption" ? item.caption : item.text}
        onChange={(event) => updateItem(item.id, { [name]: event.target.value })}
        onDoubleClick={(event) => {
          setEditing(name)
          event.currentTarget.focus()
        }}
        onBlur={() => setEditing(null)}
        onKeyDown={(event) => {
          if (event.key === "Escape") event.currentTarget.blur()
        }}
        className={cn(
          "m-1.5 min-h-0 min-w-0 flex-1 resize-none overflow-hidden rounded bg-background/50 px-1.5 py-0.5 leading-normal text-foreground outline-none placeholder:text-muted-foreground",
          "text-[length:calc(0.875rem*var(--jb-text-scale))]",
          editing === name ? "cursor-text ring-2 ring-ring/60 select-text" : "cursor-move select-none",
          invalid && "ring-2 ring-destructive",
        )}
      />
    )
  }

  return (
    <div
      role="group"
      tabIndex={0}
      aria-label={t("Box {{n}}: {{label}}", { n: index + 1, label: t(LABEL_NAMES[item.label]) })}
      data-box={item.id}
      onFocus={() => onActivate(item.id)}
      onKeyDown={onKeyDown}
      onPointerDown={(event) => {
        if (event.button !== 0) return
        const target = event.target as HTMLElement
        if (target instanceof HTMLTextAreaElement && !target.readOnly) return
        event.preventDefault()
        event.currentTarget.focus({ preventScroll: true })
        startDrag(event)
      }}
      style={{ left: `${(x1 / width) * 100}%`, top: `${(y1 / height) * 100}%`, width: `${((x2 - x1) / width) * 100}%`, height: `${((y2 - y1) / height) * 100}%` }}
      className={cn(
        "absolute flex min-h-5 min-w-5 touch-none flex-col overflow-hidden rounded border-[1.5px] border-primary/60 bg-primary/14 outline-none select-none",
        "cursor-move transition-[border-color,box-shadow] hover:border-primary focus-visible:ring-3 focus-visible:ring-ring/50",
        (active || dragging) && "z-10 border-primary shadow-lg ring-3 ring-primary/30",
      )}
    >
      <div className="flex-none truncate bg-primary px-1.5 text-[length:calc(0.6875rem*var(--jb-text-scale))] leading-normal font-medium tracking-wide text-primary-foreground uppercase">
        {t(LABEL_NAMES[item.label])}
      </div>
      {field("caption")}
      {item.label === "text" ? field("text") : null}
      <div
        aria-hidden
        className="absolute right-0 bottom-0 size-3.5 cursor-nwse-resize touch-none after:absolute after:right-0.75 after:bottom-0.75 after:size-2 after:rounded-br-[3px] after:border-r-2 after:border-b-2 after:border-primary after:opacity-60 hover:after:opacity-100"
        onPointerDown={(event) => {
          if (event.button !== 0) return
          event.stopPropagation()
          event.preventDefault()
          onActivate(item.id)
          startResize(event)
        }}
      />
    </div>
  )
}

export function LayoutCanvas({
  activeId,
  onActivate,
  missing,
  className,
}: {
  activeId: number | null
  onActivate: (id: number | null) => void
  missing: ReadonlySet<Missing>
  className?: string
}) {
  const { t } = useTranslation()
  const width = useBuilder((s) => s.width)
  const height = useBuilder((s) => s.height)
  const items = useBuilder((s) => s.items)
  const [area, setArea] = useState<HTMLDivElement | null>(null)
  const [stage, setStage] = useState<HTMLDivElement | null>(null)
  const areaSize = useElementSize(area)
  const inner = { width: areaSize.width - 2 * INSET, height: areaSize.height - 2 * INSET }
  const fitted = fitStage(inner.width, inner.height, width / height)
  // A new canvas size, or a resized area, fits the stage again (upstream's fitStage resets the view).
  const sizeKey = `${width}x${height}@${areaSize.width}x${areaSize.height}`
  const [view, setView] = useState<View>(RESET_VIEW)
  const [viewKey, setViewKey] = useState(sizeKey)
  if (viewKey !== sizeKey) {
    setViewKey(sizeKey)
    setView(RESET_VIEW)
  }
  const [panning, setPanning] = useState(false)
  // What the wheel listener (attached once per element) reads.
  const latest = useRef({ inner, fitted })
  useEffect(() => {
    latest.current = { inner, fitted }
  })

  const zoom = (factor: number, x = 0, y = 0) => setView((v) => clampView(zoomAbout(v, x, y, factor), inner, fitted))

  // The wheel zooms at the cursor (a non-passive listener, so the page does not scroll).
  useEffect(() => {
    if (!area) return
    const onWheel = (event: WheelEvent) => {
      event.preventDefault()
      const { inner, fitted } = latest.current
      const rect = area.getBoundingClientRect()
      const x = event.clientX - (rect.left + rect.width / 2)
      const y = event.clientY - (rect.top + rect.height / 2)
      setView((v) => clampView(zoomAbout(v, x, y, event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP), inner, fitted))
    }
    area.addEventListener("wheel", onWheel, { passive: false })
    return () => area.removeEventListener("wheel", onWheel)
  }, [area])

  const startPan = (event: ReactPointerEvent) => {
    const from = view
    setPanning(true)
    track(
      event,
      (dx, dy) => setView(clampView({ ...from, panX: from.panX + dx, panY: from.panY + dy }, inner, fitted)),
      () => setPanning(false),
    )
  }

  const onBackground = (target: EventTarget) => target === area || target === stage

  return (
    <div
      ref={setArea}
      data-testid="json-builder-canvas"
      className={cn("relative flex min-h-0 min-w-0 items-center justify-center overflow-hidden rounded-md bg-card", panning && "cursor-grabbing", className)}
      onPointerDown={(event) => {
        if (event.button === 1 || (event.button === 0 && view.scale > 1 && onBackground(event.target))) {
          event.preventDefault()
          startPan(event)
        } else if (event.button === 0 && onBackground(event.target)) {
          onActivate(null)
        }
      }}
      onDoubleClick={(event) => {
        if (onBackground(event.target)) setView(RESET_VIEW)
      }}
    >
      <div
        ref={setStage}
        className={cn("relative shrink-0 overflow-hidden border bg-muted/40 shadow-inner", view.scale > 1 && !panning && "cursor-grab")}
        style={
          {
            width: fitted.width,
            height: fitted.height,
            transform: `translate(${view.panX}px, ${view.panY}px) scale(${view.scale})`,
            "--jb-text-scale": String(1 / Math.max(1, view.scale)),
            backgroundImage:
              "linear-gradient(to right, color-mix(in oklch, var(--border) 70%, transparent) 1px, transparent 1px), linear-gradient(to bottom, color-mix(in oklch, var(--border) 70%, transparent) 1px, transparent 1px)",
            backgroundSize: "16px 16px",
          } as CSSProperties
        }
      >
        {items.map((item, index) => (
          <BoxView key={item.id} item={item} index={index} stage={stage} active={item.id === activeId} missing={missing} onActivate={onActivate} />
        ))}
      </div>
      <div className="absolute right-2 bottom-2 flex items-center gap-0.5 rounded-lg border bg-background/90 p-0.5 shadow-sm">
        <Button size="icon-xs" variant="ghost" aria-label={t("Zoom out")} title={t("Zoom out")} onClick={() => zoom(1 / ZOOM_STEP)}>
          <ZoomOut aria-hidden />
        </Button>
        <span className="w-10 text-center text-xs text-muted-foreground tabular-nums">{Math.round(view.scale * 100)}%</span>
        <Button size="icon-xs" variant="ghost" aria-label={t("Zoom in")} title={t("Zoom in")} onClick={() => zoom(ZOOM_STEP)}>
          <ZoomIn aria-hidden />
        </Button>
        <Button size="icon-xs" variant="ghost" aria-label={t("Fit to screen")} title={t("Fit to screen")} onClick={() => setView(RESET_VIEW)}>
          <Maximize aria-hidden />
        </Button>
      </div>
      <span className="pointer-events-none absolute top-2 left-3 text-xs text-muted-foreground tabular-nums">
        {width} × {height}
      </span>
    </div>
  )
}
