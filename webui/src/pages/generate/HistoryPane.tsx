/**
 * This session's results beside (or under) the stage. The grid (a
 * virtualized listbox, `HistoryGrid`) loads with the session's first result:
 * a session that has none shows the empty state without that code
 * (docs/research/ui-performance.md, "Bundle").
 */
import { Trash2 } from "@/components/icons"
import { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { ConfirmDialog, preloadConfirmDialog } from "@/components/ConfirmDialog"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { HistoryEmpty } from "@/features/gallery/HistoryEmpty"
import { useSession } from "@/features/gallery/session-store"
import { lazyModule, prefetch } from "@/lib/lazy-module"

const historyGrid = lazyModule(() => import("@/features/gallery/HistoryGrid"))

/** The grid once the session has results (fetched with the first one); the empty state before. */
function HistoryResults({ count, onOpen, orientation, className }: { count: number; onOpen: (key: string) => void; orientation: "grid" | "strip"; className?: string }) {
  const module = historyGrid.useModule()
  useEffect(() => {
    if (count > 0) prefetch(historyGrid)
  }, [count])
  if (count === 0) return <HistoryEmpty className={className} />
  if (!module) return <div aria-busy className={className} />
  return <module.HistoryGrid onOpen={onOpen} orientation={orientation} className={className} />
}

export function HistoryPane({ onOpen, orientation = "grid", className }: { onOpen: (key: string) => void; orientation?: "grid" | "strip"; className?: string }) {
  const { t } = useTranslation()
  const count = useSession((s) => s.items.length)
  const [confirm, setConfirm] = useState(false)
  return (
    <section aria-label={t("History")} className={cn("flex min-h-0 flex-col gap-2", className)}>
      <div className="flex items-center gap-2">
        <h2 className="text-[1.0625rem] font-semibold tracking-[-0.01em]">{t("History")}</h2>
        <span className="text-sm text-muted-foreground tabular-nums">{count}</span>
        {count > 0 ? (
          <Button
            variant="ghost"
            size="icon-xs"
            className="ml-auto text-muted-foreground"
            aria-label={t("Clear history")}
            onPointerEnter={preloadConfirmDialog}
            onFocus={preloadConfirmDialog}
            onClick={() => setConfirm(true)}
          >
            <Trash2 />
          </Button>
        ) : null}
      </div>
      <HistoryResults count={count} onOpen={onOpen} orientation={orientation} className={orientation === "grid" ? "flex-1" : undefined} />
      <ConfirmDialog
        open={confirm}
        onOpenChange={setConfirm}
        title={t("Clear the history?")}
        description={t("Only this list is cleared; the images stay in the gallery.")}
        confirmLabel={t("Clear")}
        destructive
        onConfirm={() => {
          setConfirm(false)
          useSession.getState().clear()
        }}
      />
    </section>
  )
}
