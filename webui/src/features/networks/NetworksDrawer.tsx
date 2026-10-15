/**
 * The network browser as a non-modal panel over the right side of Generate:
 * the prompts stay editable and visible while cards are clicked. Resizable
 * from its left edge (desktop), a bottom sheet on phones. Escape closes it
 * (unless a menu, dialog or search text takes the key); focus moves into the
 * search on open and back to the header button on close.
 */
import { Maximize2, X } from "lucide-react"
import { useEffect, useRef } from "react"
import { useTranslation } from "react-i18next"
import { routeHref } from "@/app/router"
import { Button } from "@/components/ui/button"
import { useIsDesktop } from "@/lib/use-media-query"
import { cn } from "@/lib/utils"
import type { GenerateTab } from "@/features/progress/run-controller"
import { type BrowserHandle, NetworkBrowser } from "./NetworkBrowser"
import { DRAWER_ID } from "./shortcut"
import { DRAWER_WIDTH, useNetworksUi } from "./store"
import { PromptTargetSwitch } from "./TargetSwitch"

function ResizeHandle() {
  const { t } = useTranslation()
  const width = useNetworksUi((s) => s.drawerWidth)
  const setWidth = useNetworksUi((s) => s.setDrawerWidth)
  const max = () => Math.max(DRAWER_WIDTH.min, window.innerWidth - 320)
  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={t("Resize the networks panel")}
      aria-valuemin={DRAWER_WIDTH.min}
      aria-valuemax={typeof window === "undefined" ? undefined : max()}
      aria-valuenow={width}
      tabIndex={0}
      onKeyDown={(event) => {
        const step = event.shiftKey ? 64 : 16
        if (event.key === "ArrowLeft") setWidth(Math.min(max(), width + step))
        else if (event.key === "ArrowRight") setWidth(width - step)
        else return
        event.preventDefault()
      }}
      onDoubleClick={() => setWidth(DRAWER_WIDTH.default)}
      onPointerDown={(event) => {
        event.preventDefault()
        const startX = event.clientX
        const startWidth = width
        const handle = event.currentTarget
        handle.setPointerCapture(event.pointerId)
        const move = (moveEvent: PointerEvent) => setWidth(Math.min(max(), startWidth + startX - moveEvent.clientX))
        const up = () => {
          handle.removeEventListener("pointermove", move)
          handle.removeEventListener("pointerup", up)
          handle.removeEventListener("pointercancel", up)
        }
        handle.addEventListener("pointermove", move)
        handle.addEventListener("pointerup", up)
        handle.addEventListener("pointercancel", up)
      }}
      className="group/resize absolute inset-y-0 -left-1.5 z-10 w-3 cursor-col-resize touch-none outline-none"
    >
      <span className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-transparent transition-colors group-hover/resize:bg-ring/60 group-focus-visible/resize:bg-ring" />
    </div>
  )
}

export default function NetworksDrawer({ tab, autoFocus, onClose, returnFocus }: { tab: GenerateTab; autoFocus: boolean; onClose: () => void; returnFocus: () => void }) {
  const { t } = useTranslation()
  const desktop = useIsDesktop()
  const width = useNetworksUi((s) => s.drawerWidth)
  const browser = useRef<BrowserHandle>(null)
  const panel = useRef<HTMLElement>(null)

  useEffect(() => {
    if (autoFocus) browser.current?.focusSearch()
  }, [autoFocus])

  const close = () => {
    onClose()
    returnFocus()
  }

  return (
    <aside
      ref={panel}
      id={DRAWER_ID}
      role="dialog"
      aria-modal="false"
      aria-label={t("Networks")}
      onKeyDown={(event) => {
        if (event.key !== "Escape" || event.defaultPrevented) return
        const target = event.target as HTMLElement
        if (target.closest("[role=menu], [role=listbox][data-slot=select-content]")) return
        event.preventDefault()
        close()
      }}
      className={cn(
        "fixed z-40 flex flex-col border-border bg-background/95 shadow-2xl backdrop-blur supports-backdrop-filter:bg-background/90",
        "animate-in duration-200 fade-in-0",
        desktop ? "top-12 right-0 bottom-0 border-l slide-in-from-right-8" : "inset-x-0 bottom-0 h-[82dvh] rounded-t-2xl border-t slide-in-from-bottom-8",
      )}
      style={desktop ? { width: `min(${width}px, calc(100vw - 4rem))` } : undefined}
    >
      {desktop ? <ResizeHandle /> : <div aria-hidden className="mx-auto mt-2 h-1 w-10 rounded-full bg-muted-foreground/30" />}
      <header className="flex items-center gap-2 px-3 pt-2">
        <h2 className="font-heading text-sm font-semibold">{t("Networks")}</h2>
        <PromptTargetSwitch className="ml-2" />
        <div className="ml-auto flex items-center gap-0.5">
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label={t("Open the Networks page")}
            title={t("Open the Networks page")}
            render={<a href={routeHref({ name: "networks" })} />}
            nativeButton={false}
          >
            <Maximize2 />
          </Button>
          <Button variant="ghost" size="icon-sm" aria-label={t("Close")} title={t("Close (Esc)")} onClick={close}>
            <X />
          </Button>
        </div>
      </header>
      <NetworkBrowser ref={browser} surface="drawer" form={tab} className="min-h-0 flex-1" />
    </aside>
  )
}
