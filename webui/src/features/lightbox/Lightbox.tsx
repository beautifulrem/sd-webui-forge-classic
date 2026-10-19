/**
 * Full-screen image viewer over the session results: wheel zoom at the
 * cursor, two-finger pinch zoom, drag to pan, double-click toggles fit ↔ actual size,
 * ←/→ (or a gamepad, `js_modal_lightbox_gamepad`) previous / next, S downloads, +/−/0/1 zoom, Esc closes. While a job
 * runs it can follow the live preview (`js_live_preview_in_modal_lightbox`).
 */
import { Dialog } from "@base-ui/react/dialog"
import { ChevronLeft, ChevronRight, Grid2x2, Maximize, Minimize, X } from "lucide-react"
import { type CSSProperties, type ReactNode, useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { useClientOption } from "@/api/app-options"
import { api } from "@/api/endpoints"
import type { FormTab } from "@/api/types"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { downloadFile } from "@/features/gallery/files"
import type { ResultItem } from "@/features/gallery/session-store"
import { ResultActions } from "@/features/send-to/ResultActions"
import { useGamepadNavigation } from "./gamepad"
import { clampPan, fitScale, initialView, pinchView, type Point, type Size, type ViewState, wheelFactor, zoomAt } from "./zoom"

type LightboxProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  items: readonly ResultItem[]
  /** Key of the shown item. */
  current: string | null
  onNavigate: (key: string) => void
  /** A live preview frame URL to show instead (follows generation). */
  live?: string | null
  tab: FormTab
  /** A side panel for the shown item (e.g. its parameters), right of the image on wide screens. */
  aside?: (item: ResultItem) => ReactNode
}

const ZOOM_STEP = 1.25

function useViewportSize(element: HTMLElement | null): Size {
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

export function Lightbox({ open, onOpenChange, items, current, onNavigate, live, tab, aside }: LightboxProps) {
  const { t } = useTranslation()
  const zoomedByDefault = useClientOption("js_modal_lightbox_initially_zoomed")
  const gamepad = useClientOption("js_modal_lightbox_gamepad")
  const gamepadRepeat = useClientOption("js_modal_lightbox_gamepad_repeat")
  // Opacity of the toolbar and the ← / → buttons until the mouse is over them (or they hold the focus).
  const toolbarOpacity = useClientOption("sd_webui_modal_lightbox_toolbar_opacity")
  const iconOpacity = useClientOption("sd_webui_modal_lightbox_icon_opacity")
  const dim = (opacity: number) => ({ "--lightbox-dim": String(Math.min(1, Math.max(0, opacity))) }) as CSSProperties
  const [viewport, setViewport] = useState<HTMLDivElement | null>(null)
  const viewportSize = useViewportSize(viewport)
  const [natural, setNatural] = useState<Size | null>(null)
  const [view, setView] = useState<ViewState>({ scale: 1, x: 0, y: 0 })
  const [tiled, setTiled] = useState(false)
  /** Pointers down on the viewport (relative to its centre). */
  const pointers = useRef(new Map<number, Point>())
  /** One pointer pans; a second one turns the gesture into a pinch. */
  const gesture = useRef<{ view: ViewState; from: Map<number, Point> } | null>(null)
  const [dragging, setDragging] = useState(false)

  const index = Math.max(0, items.findIndex((item) => item.key === current))
  const item = items[index] ?? null
  const src = live ?? (item ? api.fileUrl(item.fileId) : null)
  const fitted = natural ? fitScale(natural, viewportSize) : 1
  const isFitted = Math.abs(view.scale - fitted) < 1e-3

  // A new image starts at the configured zoom (the live preview keeps the view).
  const imageKey = live ? "live" : item?.key
  useEffect(() => {
    setNatural(null)
    setTiled(false)
  }, [imageKey])
  useEffect(() => {
    if (natural && viewportSize.width > 0) setView(initialView(natural, viewportSize, zoomedByDefault))
    // Re-fit on image or viewport change only.
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, [natural, viewportSize.width, viewportSize.height])

  const go = (delta: number) => {
    if (items.length === 0 || live) return
    const next = items[(index + delta + items.length) % items.length]
    if (next) onNavigate(next.key)
  }
  useGamepadNavigation(open && gamepad, gamepadRepeat, go)
  const setClamped = (next: ViewState) => setView(natural ? clampPan(next, natural, viewportSize) : next)
  const zoomBy = (factor: number) => setClamped(zoomAt(view, factor))
  const toggleZoom = () => setView(isFitted ? { scale: 1, x: 0, y: 0 } : { scale: fitted, x: 0, y: 0 })

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.metaKey || event.ctrlKey || event.altKey) return
    const handled: Record<string, () => void> = {
      ArrowLeft: () => go(-1),
      ArrowRight: () => go(1),
      Home: () => items[0] && onNavigate(items[0].key),
      End: () => items.at(-1) && onNavigate(items.at(-1)!.key),
      s: () => item && downloadFile(item.fileId),
      S: () => item && downloadFile(item.fileId),
      "+": () => zoomBy(ZOOM_STEP),
      "=": () => zoomBy(ZOOM_STEP),
      "-": () => zoomBy(1 / ZOOM_STEP),
      "0": () => setView({ scale: fitted, x: 0, y: 0 }),
      "1": () => setView({ scale: 1, x: 0, y: 0 }),
      t: () => setTiled((v) => !v),
    }
    const run = handled[event.key]
    if (run) {
      event.preventDefault()
      run()
    }
  }

  const relative = (event: { clientX: number; clientY: number }) => {
    const rect = viewport?.getBoundingClientRect()
    if (!rect) return { x: 0, y: 0 }
    return { x: event.clientX - rect.left - rect.width / 2, y: event.clientY - rect.top - rect.height / 2 }
  }

  const endPointer = (id: number) => {
    pointers.current.delete(id)
    gesture.current = pointers.current.size > 0 ? { view, from: new Map(pointers.current) } : null
    setDragging(pointers.current.size > 0)
  }

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Backdrop className="fixed inset-0 z-50 bg-[oklch(0.1_0_0/0.96)] duration-150 data-closed:animate-out data-closed:fade-out-0 data-open:animate-in data-open:fade-in-0" />
        <Dialog.Popup
          aria-label={t("Image viewer")}
          onKeyDown={onKeyDown}
          className="fixed inset-0 z-50 flex flex-col text-white outline-none duration-150 data-closed:animate-out data-closed:fade-out-0 data-open:animate-in data-open:fade-in-0"
        >
          <Dialog.Title className="sr-only">{t("Image viewer")}</Dialog.Title>
          <div
            style={dim(toolbarOpacity)}
            className="flex h-12 shrink-0 items-center gap-2 bg-gradient-to-b from-black/60 to-transparent px-3 opacity-(--lightbox-dim) transition-opacity focus-within:opacity-100 hover:opacity-100"
          >
            <span className="text-sm text-white/75 tabular-nums" aria-live="polite">
              {live ? t("Live preview") : items.length > 0 ? t("{{index}} of {{count}}", { index: index + 1, count: items.length }) : ""}
            </span>
            {natural ? (
              <span className="hidden text-xs text-white/50 tabular-nums sm:inline">
                {natural.width}×{natural.height} · {Math.round(view.scale * 100)}%
              </span>
            ) : null}
            <div className="ml-auto flex items-center gap-0.5">
              {!live ? <ResultActions item={item} tab={tab} tone="light" removable={false} onSent={() => onOpenChange(false)} /> : null}
              <Button
                variant="ghost"
                size="icon-sm"
                aria-pressed={tiled}
                aria-label={t("Tile preview")}
                aria-keyshortcuts="T"
                className="text-white/85 hover:bg-white/12 hover:text-white aria-pressed:bg-white/15"
                onClick={() => setTiled((v) => !v)}
              >
                <Grid2x2 />
              </Button>
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={isFitted ? t("Actual size") : t("Fit to screen")}
                className="text-white/85 hover:bg-white/12 hover:text-white"
                onClick={toggleZoom}
              >
                {isFitted ? <Maximize /> : <Minimize />}
              </Button>
              <Dialog.Close
                render={<Button variant="ghost" size="icon-sm" aria-label={t("Close")} className="text-white/85 hover:bg-white/12 hover:text-white" />}
              >
                <X />
              </Dialog.Close>
            </div>
          </div>

          <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
          <div
            ref={setViewport}
            tabIndex={-1}
            className={cn("relative min-h-0 flex-1 touch-none overflow-hidden outline-none select-none", dragging ? "cursor-grabbing" : "cursor-grab")}
            onWheel={(event) => setClamped(zoomAt(view, wheelFactor(event.deltaY, event.deltaMode), relative(event)))}
            onDoubleClick={toggleZoom}
            onPointerDown={(event) => {
              if (event.button !== 0) return
              event.currentTarget.setPointerCapture?.(event.pointerId)
              pointers.current.set(event.pointerId, relative(event))
              // Each finger landing or lifting restarts the gesture from the current view.
              gesture.current = { view, from: new Map(pointers.current) }
              setDragging(true)
            }}
            onPointerMove={(event) => {
              const start = gesture.current
              if (!start || !pointers.current.has(event.pointerId)) return
              pointers.current.set(event.pointerId, relative(event))
              const ids = [...start.from.keys()].slice(0, 2)
              const from = ids.map((id) => start.from.get(id)!)
              const to = ids.map((id) => pointers.current.get(id) ?? start.from.get(id)!)
              if (from.length === 2) setClamped(pinchView(start.view, [from[0]!, from[1]!], [to[0]!, to[1]!]))
              else if (from.length === 1) setClamped({ ...start.view, x: start.view.x + to[0]!.x - from[0]!.x, y: start.view.y + to[0]!.y - from[0]!.y })
            }}
            onPointerUp={(event) => endPointer(event.pointerId)}
            onPointerCancel={(event) => endPointer(event.pointerId)}
            onAuxClick={(event) => {
              // Middle click opens the image in a new tab, as before.
              if (event.button === 1 && item && !live) window.open(api.fileUrl(item.fileId), "_blank", "noopener")
            }}
          >
            {src && tiled ? (
              <div aria-hidden className="absolute inset-0" style={{ backgroundImage: `url("${src}")`, backgroundRepeat: "repeat", backgroundPosition: "center", backgroundSize: natural ? `${natural.width * view.scale}px` : undefined }} />
            ) : null}
            {src ? (
              <img
                key={imageKey}
                src={src}
                alt={live ? t("Live preview") : t("Generated image")}
                draggable={false}
                onLoad={(event) => {
                  const { naturalWidth: width, naturalHeight: height } = event.currentTarget
                  // Live frames keep their size: keep the object so the view is not reset per frame.
                  setNatural((prev) => (prev && prev.width === width && prev.height === height ? prev : { width, height }))
                }}
                className={cn("absolute top-1/2 left-1/2 max-w-none origin-center", tiled && "opacity-0", !natural && "opacity-0")}
                style={{
                  width: natural?.width,
                  height: natural?.height,
                  transform: `translate(-50%, -50%) translate(${view.x}px, ${view.y}px) scale(${view.scale})`,
                  imageRendering: view.scale > 2 ? "pixelated" : undefined,
                }}
              />
            ) : (
              <p className="absolute inset-0 grid place-items-center text-sm text-white/60">{t("No image")}</p>
            )}
            {!live && items.length > 1 ? (
              <>
                <Button
                  variant="ghost"
                  size="icon-lg"
                  aria-label={t("Previous image")}
                  aria-keyshortcuts="ArrowLeft"
                  onPointerDown={(e) => e.stopPropagation()}
                  onClick={() => go(-1)}
                  style={dim(iconOpacity)}
                  className="absolute top-1/2 left-2 -translate-y-1/2 rounded-full bg-black/35 text-white opacity-(--lightbox-dim) transition-opacity hover:bg-black/55 hover:text-white hover:opacity-100 focus-visible:opacity-100"
                >
                  <ChevronLeft />
                </Button>
                <Button
                  variant="ghost"
                  size="icon-lg"
                  aria-label={t("Next image")}
                  aria-keyshortcuts="ArrowRight"
                  onPointerDown={(e) => e.stopPropagation()}
                  onClick={() => go(1)}
                  style={dim(iconOpacity)}
                  className="absolute top-1/2 right-2 -translate-y-1/2 rounded-full bg-black/35 text-white opacity-(--lightbox-dim) transition-opacity hover:bg-black/55 hover:text-white hover:opacity-100 focus-visible:opacity-100"
                >
                  <ChevronRight />
                </Button>
              </>
            ) : null}
          </div>
          {aside && item && !live ? (
            <aside
              aria-label={t("Image details")}
              className="max-h-[40%] shrink-0 overflow-y-auto border-t border-white/10 bg-black/40 p-3 lg:max-h-none lg:w-80 lg:border-t-0 lg:border-l xl:w-96"
            >
              {aside(item)}
            </aside>
          ) : null}
          </div>
        </Dialog.Popup>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
