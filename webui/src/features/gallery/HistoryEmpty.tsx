import { History } from "@/components/icons"
import { useTranslation } from "react-i18next"
import { cn } from "@/lib/utils"

/** The history before the session's first result (what `HistoryGrid` shows for an empty session). */
export function HistoryEmpty({ className }: { className?: string }) {
  const { t } = useTranslation()
  return (
    <div className={cn("flex flex-col items-center justify-center gap-2 p-6 text-center text-xs text-muted-foreground", className)}>
      <History className="size-6 opacity-50" aria-hidden />
      <p>{t("Results of this session appear here")}</p>
    </div>
  )
}
