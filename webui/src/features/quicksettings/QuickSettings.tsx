/**
 * Header button for the pinned settings (`quicksettings_list`): a popover
 * with each setting rendered by the schema renderer and applied as it
 * changes. Hidden while nothing is pinned; pin settings from the Settings
 * page. The model bar next to it covers checkpoint, VAE / TE, dtype, preset.
 */
import { Loader2, Settings2, Zap } from "@/components/icons"
import { lazy, Suspense } from "react"
import { useTranslation } from "react-i18next"
import { optionsQuery, queryClient } from "@/api/queries"
import { navigate } from "@/app/router"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { usePopupTooltip } from "@/components/popup-tooltip"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { useQuickKeys } from "./pinned"

const QuickSettingsPanel = lazy(() => import("./QuickSettingsPanel"))
const preloadPanel = () => {
  void import("./QuickSettingsPanel")
  void queryClient.prefetchQuery(optionsQuery)
}

export default function QuickSettings() {
  const { t } = useTranslation()
  const pinned = useQuickKeys()
  const tip = usePopupTooltip()
  const open = tip.popupOpen
  if (pinned.length === 0) return null
  return (
    <Popover open={open} onOpenChange={tip.onPopupOpenChange}>
      <Tooltip {...tip.tooltip}>
        <TooltipTrigger
          render={
            <PopoverTrigger
              render={
                <Button
                  variant="ghost"
                  size="sm"
                  className="gap-1 px-2"
                  aria-label={t("Quick settings")}
                  onPointerEnter={preloadPanel}
                  onFocus={preloadPanel}
                  {...tip.trigger}
                />
              }
            />
          }
        >
          <Zap aria-hidden />
          <Badge variant="secondary" className="h-4 min-w-4 px-1 text-[0.65rem] tabular-nums">
            {pinned.length}
          </Badge>
        </TooltipTrigger>
        <TooltipContent>{t("Quick settings")}</TooltipContent>
      </Tooltip>
      <PopoverContent aria-label={t("Quick settings")} align="end" className="w-[min(24rem,calc(100vw-2rem))] gap-2 p-2">
        <div className="flex items-center justify-between gap-2 px-2 pt-1">
          <p className="text-sm font-medium">{t("Quick settings")}</p>
          <Button
            variant="ghost"
            size="xs"
            onClick={() => {
              tip.onPopupOpenChange(false)
              navigate({ name: "settings", query: new URLSearchParams({ filter: "pinned" }) })
            }}
          >
            <Settings2 aria-hidden />
            {t("All settings")}
          </Button>
        </div>
        {open ? (
          <Suspense
            fallback={
              <div className="grid place-items-center py-8" aria-busy="true">
                <Loader2 className="size-5 animate-spin text-muted-foreground" aria-label={t("Loading…")} />
              </div>
            }
          >
            <QuickSettingsPanel keys={pinned} />
          </Suspense>
        ) : null}
        <p className="px-2 pb-1 text-xs text-muted-foreground">{t("Changes apply at once. Pin more from Settings.")}</p>
      </PopoverContent>
    </Popover>
  )
}
