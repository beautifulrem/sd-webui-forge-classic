/** This session's results beside (or under) the stage. */
import { Trash2 } from "lucide-react"
import { useState } from "react"
import { useTranslation } from "react-i18next"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { HistoryGrid } from "@/features/gallery/HistoryGrid"
import { useSession } from "@/features/gallery/session-store"

export function HistoryPane({ onOpen, orientation = "grid", className }: { onOpen: (key: string) => void; orientation?: "grid" | "strip"; className?: string }) {
  const { t } = useTranslation()
  const count = useSession((s) => s.items.length)
  const [confirm, setConfirm] = useState(false)
  return (
    <section aria-label={t("History")} className={cn("flex min-h-0 flex-col gap-2", className)}>
      <div className="flex items-center gap-2">
        <h2 className="text-sm font-medium">{t("History")}</h2>
        <Badge variant="secondary" className="tabular-nums">
          {count}
        </Badge>
        {count > 0 ? (
          <Button variant="ghost" size="icon-xs" className="ml-auto text-muted-foreground" aria-label={t("Clear history")} onClick={() => setConfirm(true)}>
            <Trash2 />
          </Button>
        ) : null}
      </div>
      <HistoryGrid onOpen={onOpen} orientation={orientation} className={orientation === "grid" ? "flex-1" : undefined} />
      <AlertDialog open={confirm} onOpenChange={setConfirm}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("Clear the history?")}</AlertDialogTitle>
            <AlertDialogDescription>{t("Only this list is cleared; the images stay in the gallery.")}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("Cancel")}</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                setConfirm(false)
                useSession.getState().clear()
              }}
            >
              {t("Clear")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </section>
  )
}
