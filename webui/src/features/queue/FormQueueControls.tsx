/**
 * Queue controls shown with a form's run controls: "Add to queue" (queue the
 * form as a job without waiting for it here; Ctrl/Cmd+Shift+Enter, see the
 * pages' shortcuts) and, while a queued job is
 * being edited, the banner with "Update job" / "Cancel".
 */
import { ListPlus, Loader2, PencilLine, X } from "@/components/icons"
import { useState } from "react"
import { useTranslation } from "react-i18next"
import type { FormTab } from "@/api/types"
import { formatShortcut } from "@/app/commands"
import { Button } from "@/components/ui/button"
import { Kbd } from "@/components/ui/kbd"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { cancelEdit, enqueueForm, saveEdit, useQueueEdit } from "./edit-session"

export function EnqueueButton({ tab, className, size = "icon-lg" }: { tab: FormTab; className?: string; size?: "icon-lg" | "icon" }) {
  const { t } = useTranslation()
  const [busy, setBusy] = useState(false)
  const label = t("Add to queue")
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <Button
            variant="outline"
            size={size}
            className={cn(size === "icon-lg" && "size-11", className)}
            aria-label={label}
            disabled={busy}
            onClick={() => {
              setBusy(true)
              void enqueueForm(tab).finally(() => setBusy(false))
            }}
          />
        }
      >
        {busy ? <Loader2 className="animate-spin" aria-hidden /> : <ListPlus />}
      </TooltipTrigger>
      <TooltipContent>
        {t("Add to queue: runs after the queued jobs, results go to the gallery")}
        <Kbd className="ml-1.5">{formatShortcut("Mod+Shift+Enter")}</Kbd>
      </TooltipContent>
    </Tooltip>
  )
}

export function QueueEditBanner({ tab, className }: { tab: FormTab; className?: string }) {
  const { t } = useTranslation()
  const session = useQueueEdit((s) => s.sessions[tab])
  const saving = useQueueEdit((s) => s.saving[tab] ?? false)
  if (!session) return null
  return (
    <div role="status" className={cn("flex items-center gap-2 rounded-lg border border-primary/30 bg-primary/8 px-2.5 py-1.5 text-sm", className)}>
      <PencilLine className="size-4 shrink-0 text-primary" aria-hidden />
      <div className="min-w-0 flex-1">
        <p className="font-medium">{t("Editing queued job #{{seq}}", { seq: session.seq })}</p>
        {session.title ? <p className="truncate text-xs text-muted-foreground">{session.title}</p> : null}
      </div>
      <Button size="sm" onClick={() => void saveEdit(tab)} disabled={saving}>
        {saving ? <Loader2 className="animate-spin" aria-hidden /> : null}
        {t("Update job")}
      </Button>
      <Button variant="ghost" size="icon-sm" aria-label={t("Stop editing")} title={t("Stop editing")} onClick={() => cancelEdit(tab)}>
        <X />
      </Button>
    </div>
  )
}
