/**
 * Import the queues of the extensions the Queue replaces (Agent Scheduler,
 * Prompt Queue): what was found, one click per source, then the report —
 * counts and every task that was not imported exactly, with why.
 *
 * Closing the dialog while an import runs does not lose it: a toast offers
 * the report when it is done, and the report (with "Resume the queue" when
 * the import paused it) stays until "Done".
 */
import { useMutation, useQuery } from "@tanstack/react-query"
import { CheckCircle2, CircleAlert, DatabaseZap, Loader2, Play } from "lucide-react"
import { useEffect, useRef } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import type { ImportItem, ImportReport, ImportSource } from "@/api/types"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Skeleton } from "@/components/ui/skeleton"
import { errorText } from "@/features/errors/error-text"
import { noteText } from "./notes"
import { act, importSourcesQuery, refreshQueue } from "./queries"

const OUTCOMES: Record<ImportItem["outcome"], { label: string; variant: "default" | "secondary" | "outline" | "destructive" }> = {
  queued: { label: "Queued", variant: "default" },
  history: { label: "History", variant: "secondary" },
  refused: { label: "Not replayable", variant: "outline" },
  invalid: { label: "Needs editing", variant: "outline" },
  skipped: { label: "Skipped", variant: "outline" },
  error: { label: "Error", variant: "destructive" },
}

function Report({ report }: { report: ImportReport }) {
  const { t } = useTranslation()
  const counts: [string, number][] = [
    [t("Queued"), report.queued],
    [t("History"), report.history],
    [t("Not replayable"), report.refused],
    [t("Needs editing"), report.invalid],
    [t("Skipped"), report.skipped],
    [t("Errors"), report.errors],
  ]
  return (
    <div className="flex min-h-0 flex-col gap-3">
      <dl className="grid grid-cols-3 gap-2 sm:grid-cols-6">
        {counts.map(([label, value]) => (
          <div key={label} className="rounded-lg bg-muted px-2 py-1.5 text-center">
            <dd className="text-lg font-semibold tabular-nums">{value}</dd>
            <dt className="text-[0.7rem] text-muted-foreground">{label}</dt>
          </div>
        ))}
      </dl>
      {report.paused ? (
        <p className="flex items-start gap-2 rounded-lg border border-primary/30 bg-primary/8 px-3 py-2 text-sm">
          <CircleAlert className="mt-0.5 size-4 shrink-0 text-primary" aria-hidden />
          {t("The queue was paused so you can review the imported jobs first.")}
        </p>
      ) : null}
      {report.items.length > 0 ? (
        <ul aria-label={t("Imported tasks")} className="max-h-72 min-h-0 overflow-y-auto rounded-lg border">
          {report.items.map((item) => (
            <li key={item.source_id} className="flex flex-col gap-0.5 border-b px-3 py-2 last:border-0">
              <div className="flex items-center gap-2">
                <span className="min-w-0 flex-1 truncate text-sm">{item.title}</span>
                <Badge variant={OUTCOMES[item.outcome].variant}>{t(OUTCOMES[item.outcome].label)}</Badge>
              </div>
              {item.notes.length > 0 ? (
                <ul className="list-disc pl-5 text-xs text-muted-foreground">
                  {item.notes.map((note, index) => (
                    <li key={index}>{noteText(note)}</li>
                  ))}
                </ul>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  )
}

export function ImportDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const { t } = useTranslation()
  const sources = useQuery({ ...importSourcesQuery, enabled: open })
  const openRef = useRef(open)
  useEffect(() => {
    openRef.current = open
  }, [open])
  const importError = (error: unknown) =>
    errorText(error, {
      404: t("Nothing to import was found on this machine."),
      422: t("The file could not be read: it is not an Agent Scheduler database or a Prompt Queue file."),
    })
  const run = useMutation({
    mutationFn: (source: ImportSource) => api.queue.importFrom(source),
    onSettled: refreshQueue,
    onSuccess: (report) => {
      if (openRef.current) return
      // Finished after the dialog was closed: offer the report.
      toast.success(t("The import is done"), {
        description: report.paused ? t("The queue was paused so you can review the imported jobs first.") : undefined,
        action: { label: t("View report"), onClick: () => onOpenChange(true) },
      })
    },
    onError: (error) => {
      if (!openRef.current) toast.error(t("The import failed"), { description: importError(error) })
    },
  })
  const report = run.data
  const finish = () => {
    onOpenChange(false)
    run.reset()
  }
  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        onOpenChange(next)
        // Keep a running import, and a report that still asks to resume the queue.
        if (!next && !run.isPending && !report?.paused) run.reset()
      }}
    >
      <DialogContent className="flex max-h-[90dvh] flex-col sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>{t("Import queues")}</DialogTitle>
          <DialogDescription>
            {t("Bring the tasks of the Agent Scheduler and the Prompt Queue into this queue. Their data is only read; importing again skips what was imported before.")}
          </DialogDescription>
        </DialogHeader>
        {report ? (
          <Report report={report} />
        ) : sources.isPending ? (
          <Skeleton className="h-28 w-full" />
        ) : (
          <ul className="flex flex-col gap-2">
            {(sources.data ?? []).map((source) => (
              <li key={source.id} className="flex items-center gap-3 rounded-lg border p-3">
                <DatabaseZap className="size-5 shrink-0 text-muted-foreground" aria-hidden />
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium">{t(source.label)}</p>
                  <p className="truncate text-xs text-muted-foreground" title={source.path ?? undefined}>
                    {!source.available
                      ? t("Nothing found on this machine")
                      : source.path
                        ? t("{{count}} records · {{path}}", { count: source.count, path: source.path })
                        : t("Records: {{count}}", { count: source.count })}
                  </p>
                </div>
                <Button size="sm" disabled={!source.available || run.isPending} onClick={() => run.mutate(source.id)}>
                  {run.isPending && run.variables === source.id ? <Loader2 className="animate-spin" aria-hidden /> : null}
                  {t("Import")}
                </Button>
              </li>
            ))}
          </ul>
        )}
        {run.isPending ? (
          <p role="status" className="text-xs text-muted-foreground">
            {t("Importing… You can close this dialog: the report is offered when the import is done.")}
          </p>
        ) : null}
        {run.isError ? (
          <p role="alert" className="text-sm text-destructive">
            {importError(run.error)}
          </p>
        ) : null}
        <DialogFooter>
          {report?.paused ? (
            <Button
              variant="secondary"
              onClick={() => {
                void act(() => api.queue.resume(), t("Could not resume the queue"))
                finish()
              }}
            >
              <Play aria-hidden />
              {t("Resume the queue")}
            </Button>
          ) : null}
          <Button onClick={finish}>
            {report ? <CheckCircle2 aria-hidden /> : null}
            {report ? t("Done") : t("Close")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
