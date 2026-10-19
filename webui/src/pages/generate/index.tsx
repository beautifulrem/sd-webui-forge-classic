/**
 * Generate (txt2img and img2img share this view; the mode switch sits in the
 * header). Desktop: three resizable, collapsible panes — parameters with the
 * sticky run controls | stage | history. Tablets: parameters | stage with a
 * history strip. Phones: one column with the run controls stuck to the bottom.
 */
import { PanelLeftClose, PanelLeftOpen, PanelRightClose, PanelRightOpen } from "lucide-react"
import { lazy, Suspense, useState } from "react"
import { useTranslation } from "react-i18next"
import { useDefaultLayout, usePanelRef } from "react-resizable-panels"
import { useClientOption } from "@/api/app-options"
import { api } from "@/api/endpoints"
import { useSchema } from "@/api/queries"
import { useWs } from "@/api/ws"
import { HeaderSlot } from "@/app/header-slot"
import { type Route, routeHref } from "@/app/router"
import { Button } from "@/components/ui/button"
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable"
import { Skeleton } from "@/components/ui/skeleton"
import { useIsDesktop, useMediaQuery } from "@/lib/use-media-query"
import { cn } from "@/lib/utils"
import { selectedItem, useSession } from "@/features/gallery/session-store"
import type { GenerateTab } from "@/features/progress/run-controller"
import { NetworksDrawerHost } from "@/features/networks/NetworksDrawerHost"
import { GenerateBar } from "./GenerateBar"
import { HistoryPane } from "./HistoryPane"
import { useOwnRunningJob } from "./hooks"
import { ParamsPane } from "./ParamsPane"
import { ResetFormDialog } from "./ResetFormDialog"
import { useGenerateCommands, useGenerateRuntime } from "./runtime"
import { useGenerateShortcuts } from "./shortcuts"
import { Stage } from "./Stage"

// The viewer's chunk loads on first open.
const Lightbox = lazy(() => import("@/features/lightbox/Lightbox").then((m) => ({ default: m.Lightbox })))

/** txt2img ↔ img2img: two links styled as a segmented control (they are navigation). */
function ModeSwitch({ tab }: { tab: GenerateTab }) {
  const { t } = useTranslation()
  const modes: { value: GenerateTab; label: string }[] = [
    { value: "txt2img", label: t("txt2img") },
    { value: "img2img", label: t("img2img") },
  ]
  return (
    <nav aria-label={t("Mode")} className="inline-flex shrink-0 items-center gap-0.5 rounded-lg bg-muted p-0.5">
      {modes.map((mode) => (
        <a
          key={mode.value}
          href={routeHref({ name: "generate", tab: mode.value })}
          aria-current={mode.value === tab ? "page" : undefined}
          className={cn(
            "flex h-6 items-center rounded-md px-2.5 text-[0.8rem] font-medium whitespace-nowrap text-muted-foreground transition-colors outline-none select-none",
            "hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/60",
            "aria-[current=page]:bg-background aria-[current=page]:text-foreground aria-[current=page]:shadow-sm dark:aria-[current=page]:bg-accent",
          )}
        >
          {mode.label}
        </a>
      ))}
    </nav>
  )
}

/** The lightbox over the session results; subscribes to the live preview itself (no page re-render per frame). */
function ViewerHost({ open, onOpenChange, tab }: { open: boolean; onOpenChange: (open: boolean) => void; tab: GenerateTab }) {
  const liveInViewer = useClientOption("js_live_preview_in_modal_lightbox")
  const items = useSession((s) => s.items)
  const current = useSession((s) => selectedItem(s)?.key ?? null)
  const follow = useSession((s) => s.follow)
  const job = useOwnRunningJob()
  const preview = useWs((s) => (job && s.preview && (job.seq === undefined || s.preview.jobSeq === job.seq) ? s.preview.url : null))
  return (
    <Suspense fallback={null}>
      <Lightbox
        open={open}
        onOpenChange={onOpenChange}
        items={items}
        current={current}
        live={liveInViewer && follow ? preview : null}
        tab={tab}
        upscalable
        onNavigate={(key) => {
          useSession.getState().select(key)
          useSession.getState().setFollow(false)
        }}
      />
    </Suspense>
  )
}

function useViewer(tab: GenerateTab) {
  const [open, setOpen] = useState(false)
  // The viewer's chunk is fetched on first open and stays mounted (close animation).
  const [used, setUsed] = useState(false)
  if (open && !used) setUsed(true)
  const lightboxEnabled = useClientOption("js_modal_lightbox")

  const openViewer = (key?: string) => {
    const session = useSession.getState()
    if (key) {
      session.select(key)
      session.setFollow(false)
    }
    if (lightboxEnabled) {
      setOpen(true)
      return
    }
    const item = selectedItem(useSession.getState())
    if (item) window.open(api.fileUrl(item.fileId), "_blank", "noopener")
  }

  const viewer = used ? <ViewerHost open={open} onOpenChange={setOpen} tab={tab} /> : null
  return { openViewer, viewer }
}

function ParamsColumn({ tab, sticky, showModelBar }: { tab: GenerateTab; sticky: "top" | "bottom"; showModelBar?: boolean }) {
  const { data: schema } = useSchema(tab)
  const bar = (
    <div
      className={
        sticky === "top"
          ? "sticky top-0 z-10 border-b bg-background/85 p-3 backdrop-blur supports-backdrop-filter:bg-background/70"
          : "sticky bottom-0 z-10 border-t bg-background/90 p-3 backdrop-blur supports-backdrop-filter:bg-background/75"
      }
    >
      <GenerateBar tab={tab} />
    </div>
  )
  return (
    <>
      {sticky === "top" ? bar : null}
      <div className="p-3">
        <ParamsPane schema={schema} tab={tab} showModelBar={showModelBar} />
      </div>
      {sticky === "bottom" ? bar : null}
    </>
  )
}

function ParamsFallback() {
  return (
    <div className="flex flex-col gap-3 p-3" aria-busy>
      <Skeleton className="h-10 rounded-lg" />
      <Skeleton className="h-28 rounded-xl" />
      <Skeleton className="h-40 rounded-xl" />
    </div>
  )
}

function PaneToggle({ collapsed, side, onToggle }: { collapsed: boolean; side: "left" | "right"; onToggle: () => void }) {
  const { t } = useTranslation()
  const label = side === "left" ? (collapsed ? t("Show parameters") : t("Hide parameters")) : collapsed ? t("Show history") : t("Hide history")
  const Icon = side === "left" ? (collapsed ? PanelLeftOpen : PanelLeftClose) : collapsed ? PanelRightOpen : PanelRightClose
  return (
    <Button variant="ghost" size="icon-sm" aria-label={label} title={label} aria-pressed={!collapsed} onClick={onToggle}>
      <Icon />
    </Button>
  )
}

function DesktopLayout({ tab, openViewer, wide }: { tab: GenerateTab; openViewer: (key?: string) => void; wide: boolean }) {
  const { t } = useTranslation()
  const layout = useDefaultLayout({ id: wide ? "remi-generate-3" : "remi-generate-2", panelIds: wide ? ["params", "stage", "history"] : ["params", "stage"] })
  const paramsRef = usePanelRef()
  const historyRef = usePanelRef()
  const [paramsCollapsed, setParamsCollapsed] = useState(false)
  const [historyCollapsed, setHistoryCollapsed] = useState(false)
  const toggle = (ref: typeof paramsRef) => {
    const panel = ref.current
    if (!panel) return
    if (panel.isCollapsed()) panel.expand()
    else panel.collapse()
  }

  return (
    <ResizablePanelGroup orientation="horizontal" defaultLayout={layout.defaultLayout} onLayoutChanged={layout.onLayoutChanged}>
      <ResizablePanel
        id="params"
        panelRef={paramsRef}
        defaultSize={wide ? "34" : "44"}
        minSize={300}
        maxSize="60"
        collapsible
        collapsedSize={0}
        onResize={(size) => setParamsCollapsed(size.inPixels < 1)}
      >
        <div className="relative h-full overflow-y-auto overscroll-contain">
          <Suspense fallback={<ParamsFallback />}>
            <ParamsColumn tab={tab} sticky="top" />
          </Suspense>
        </div>
      </ResizablePanel>
      <ResizableHandle withHandle aria-label={t("Resize the parameters panel")} />
      <ResizablePanel id="stage" minSize={320}>
        <div className="flex h-full flex-col gap-2 p-3">
          <Stage
            tab={tab}
            onOpenViewer={() => openViewer()}
            className="min-h-0 flex-1"
            leading={<PaneToggle side="left" collapsed={paramsCollapsed} onToggle={() => toggle(paramsRef)} />}
            trailing={wide ? <PaneToggle side="right" collapsed={historyCollapsed} onToggle={() => toggle(historyRef)} /> : null}
            toolbar={paramsCollapsed ? <GenerateBar tab={tab} className="min-w-0 flex-1" /> : null}
          />
          {wide ? null : <HistoryPane onOpen={openViewer} orientation="strip" />}
        </div>
      </ResizablePanel>
      {wide ? (
        <>
          <ResizableHandle withHandle aria-label={t("Resize the history panel")} />
          <ResizablePanel
            id="history"
            panelRef={historyRef}
            defaultSize="18"
            minSize={150}
            maxSize="35"
            collapsible
            collapsedSize={0}
            onResize={(size) => setHistoryCollapsed(size.inPixels < 1)}
          >
            <HistoryPane onOpen={openViewer} className="h-full p-3" />
          </ResizablePanel>
        </>
      ) : null}
    </ResizablePanelGroup>
  )
}

/** Window effects that follow every progress step (title, wake lock), kept out of the page's render. */
function RuntimeEffects({ tab }: { tab: GenerateTab }) {
  useGenerateRuntime(tab)
  return null
}

export default function GeneratePage({ route }: { route: Extract<Route, { name: "generate" }> }) {
  const tab = route.tab
  const desktop = useIsDesktop()
  const wide = useMediaQuery("(min-width: 72rem)")
  const { openViewer, viewer } = useViewer(tab)
  useGenerateShortcuts(tab)
  useGenerateCommands(tab, openViewer)

  return (
    <>
      <RuntimeEffects tab={tab} />
      <HeaderSlot>
        <ModeSwitch tab={tab} />
      </HeaderSlot>
      <NetworksDrawerHost tab={tab} />
      {desktop ? (
        <DesktopLayout key={wide ? "wide" : "medium"} tab={tab} openViewer={openViewer} wide={wide} />
      ) : (
        <div className="relative h-full overflow-y-auto overscroll-contain">
          <div className="flex flex-col gap-3 p-3 pb-0">
            <Stage tab={tab} onOpenViewer={() => openViewer()} className="h-[min(70vh,36rem)]" />
            <HistoryPane onOpen={openViewer} orientation="strip" />
          </div>
          <Suspense fallback={<ParamsFallback />}>
            <ParamsColumn tab={tab} sticky="bottom" showModelBar />
          </Suspense>
        </div>
      )}
      {viewer}
      <ResetFormDialog />
    </>
  )
}
