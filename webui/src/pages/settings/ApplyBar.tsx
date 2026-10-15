/** The unsaved-changes bar (Apply / Discard / Review) and the "takes effect after a reload/restart" notice. */
import { Loader2, Power, RefreshCw, X } from "lucide-react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { Kbd } from "@/components/ui/kbd"
import { formatShortcut } from "@/app/commands"
import { requestServerCommand } from "@/features/server-control/server-control"
import { useSettingsEditor } from "./settings-store"

export function ApplyBar(props: { count: number; onApply: () => void; onDiscard: () => void; onReview: () => void; reviewing: boolean }) {
  const { t } = useTranslation()
  const applying = useSettingsEditor((s) => s.applying)
  if (props.count === 0 && !applying) return null
  return (
    <div
      role="region"
      aria-label={t("Unsaved changes")}
      className="pointer-events-none absolute inset-x-0 bottom-0 z-10 flex justify-center px-3 pb-3 sm:pb-4"
    >
      <div className="pointer-events-auto flex w-full max-w-2xl flex-wrap items-center gap-2 rounded-xl border bg-popover/95 px-3 py-2 shadow-lg backdrop-blur supports-backdrop-filter:bg-popover/85 animate-in fade-in slide-in-from-bottom-2">
        <p className="min-w-0 flex-1 text-sm" aria-live="polite">
          {applying ? t("Applying…") : t("Unsaved changes: {{count}}", { count: props.count })}
        </p>
        <Button variant="ghost" size="sm" onClick={props.onReview} aria-pressed={props.reviewing} disabled={applying}>
          {props.reviewing ? t("Show all") : t("Review")}
        </Button>
        <Button variant="outline" size="sm" onClick={props.onDiscard} disabled={applying}>
          {t("Discard")}
        </Button>
        <Button size="sm" onClick={props.onApply} disabled={applying || props.count === 0} className="gap-1.5">
          {applying ? <Loader2 className="animate-spin" aria-hidden /> : null}
          {t("Apply settings")}
          <Kbd className="hidden bg-primary-foreground/15 text-primary-foreground sm:inline-flex">{formatShortcut("Mod+S")}</Kbd>
        </Button>
      </div>
    </div>
  )
}

/** Shown after applying settings that need a reload or restart. */
export function EffectNotice() {
  const { t } = useTranslation()
  const notice = useSettingsEditor((s) => s.notice)
  const dismiss = useSettingsEditor((s) => s.dismissNotice)
  const restart = notice.restart.length > 0
  if (!restart && notice.reload.length === 0) return null
  const keys = [...notice.restart, ...notice.reload]
  return (
    <div role="status" className="flex flex-wrap items-center gap-2 border-b bg-warning/10 px-4 py-2 text-sm sm:px-6">
      {restart ? <Power className="size-4 shrink-0 text-warning" aria-hidden /> : <RefreshCw className="size-4 shrink-0 text-warning" aria-hidden />}
      <p className="min-w-0 flex-1">
        {restart ? t("Some applied settings take effect after a restart.") : t("Some applied settings take effect after Reload UI.")}{" "}
        <span className="font-mono text-xs text-muted-foreground">{keys.slice(0, 4).join(", ")}{keys.length > 4 ? " …" : ""}</span>
      </p>
      <Button size="sm" variant="outline" onClick={() => requestServerCommand(restart ? "restart" : "reload")}>
        {restart ? t("Restart server") : t("Reload UI")}
      </Button>
      <Button size="icon-sm" variant="ghost" onClick={dismiss} aria-label={t("Dismiss")}>
        <X />
      </Button>
    </div>
  )
}
