import { Palette, Search } from "lucide-react"
import { lazy, Suspense } from "react"
import { useTranslation } from "react-i18next"
import type { LocaleInfo } from "@/api/types"
import { progressFraction, useProgress } from "@/api/progress"
import { useWs, type WsStatus } from "@/api/ws"
import { Button } from "@/components/ui/button"
import { Kbd } from "@/components/ui/kbd"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { formatShortcut, useCommands } from "./commands"
import { useHeaderSlotTarget } from "./header-slot-store"

// Menus and the model picker pull in Base UI's menu/combobox; load them after first paint.
const AppearanceMenu = lazy(() => import("./AppearanceMenu"))
const ModelQuickBar = lazy(() => import("./ModelQuickBar"))
const QuickSettings = lazy(() => import("@/features/quicksettings/QuickSettings"))

const STATUS_LABEL: Record<WsStatus, string> = {
  idle: "Connecting…",
  connecting: "Connecting…",
  open: "Connected",
  reconnecting: "Reconnecting…",
  closed: "Disconnected",
}

/**
 * Connection state: a dot with a hover tooltip for pointer users and a
 * polite, non-focusable live region for screen readers (no extra tab stop).
 */
function ConnectionDot() {
  const { t } = useTranslation()
  const status = useWs((s) => s.status)
  const label = t(STATUS_LABEL[status])
  return (
    <>
      <span role="status" className="sr-only">
        {label}
      </span>
      <Tooltip>
        <TooltipTrigger render={<span aria-hidden className="grid size-8 place-items-center rounded-lg" />}>
          <span
            className={cn(
              "size-2 rounded-full",
              status === "open" && "bg-success",
              (status === "connecting" || status === "idle" || status === "reconnecting") && "animate-pulse bg-warning",
              status === "closed" && "bg-destructive",
            )}
          />
        </TooltipTrigger>
        <TooltipContent>{label}</TooltipContent>
      </Tooltip>
    </>
  )
}

/** Thin progress line along the header's bottom edge while a job runs. */
function HeaderProgress() {
  const { t } = useTranslation()
  const running = useProgress((s) => s.running)
  const fraction = progressFraction(running)
  if (fraction === null) return null
  return (
    <div
      role="progressbar"
      aria-label={t("Generation progress")}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(fraction * 100)}
      className="absolute inset-x-0 -bottom-px h-0.5 overflow-hidden"
    >
      <div
        className="h-full origin-left bg-primary transition-transform duration-300 ease-out"
        style={{ transform: `scaleX(${fraction})` }}
      />
    </div>
  )
}

export function Header({ title, locales }: { title: string; locales: LocaleInfo[] }) {
  const { t } = useTranslation()
  const setSlot = useHeaderSlotTarget((s) => s.setElement)
  const openPalette = () => useCommands.getState().setPalette(true)

  return (
    <header className="relative flex h-12 shrink-0 items-center gap-2 border-b bg-background/80 px-3 backdrop-blur supports-backdrop-filter:bg-background/65 md:px-4">
      <h1 className="shrink-0 truncate font-heading text-sm font-semibold tracking-tight">{title}</h1>
      <div ref={setSlot} className="flex min-w-0 items-center gap-2" />
      <div className="ml-auto flex min-w-0 items-center gap-1">
        <div className="hidden min-w-0 sm:block">
          <Suspense fallback={<div aria-hidden className="h-7 w-[min(22rem,40vw)] rounded-lg border border-input" />}>
            <ModelQuickBar />
          </Suspense>
        </div>
        <Suspense fallback={null}>
          <QuickSettings />
        </Suspense>
        <Button
          variant="outline"
          size="sm"
          onClick={openPalette}
          className="hidden gap-2 pr-1.5 text-muted-foreground lg:inline-flex"
        >
          <Search aria-hidden />
          <span>{t("Search")}</span>
          <Kbd>{formatShortcut("Mod+K")}</Kbd>
        </Button>
        <Button variant="ghost" size="icon" onClick={openPalette} className="lg:hidden" aria-label={t("Search")}>
          <Search />
        </Button>
        <ConnectionDot />
        <Suspense
          fallback={
            <Button variant="ghost" size="icon" disabled aria-label={t("Appearance")}>
              <Palette />
            </Button>
          }
        >
          <AppearanceMenu locales={locales} />
        </Suspense>
      </div>
      <HeaderProgress />
    </header>
  )
}
