/** Queue and history (GET /jobs); live updates arrive through the WebSocket bridge. */
import { useMutation } from "@tanstack/react-query"
import { useVirtualizer } from "@tanstack/react-virtual"
import { useEffect, useRef } from "react"
import { Ban, ListOrdered, Loader2, RotateCw, Trash2 } from "lucide-react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import { useProgress } from "@/api/progress"
import { keys, queryClient, useJobs } from "@/api/queries"
import type { Job, JobStatus } from "@/api/types"
import { PagePlaceholder } from "@/components/PagePlaceholder"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"

const STATUS: Record<JobStatus, { label: string; variant: "default" | "secondary" | "destructive" | "outline" }> = {
  queued: { label: "Queued", variant: "outline" },
  running: { label: "Running", variant: "default" },
  completed: { label: "Completed", variant: "secondary" },
  failed: { label: "Failed", variant: "destructive" },
  cancelled: { label: "Cancelled", variant: "outline" },
  interrupted: { label: "Interrupted", variant: "outline" },
}

function JobAction(props: { label: string; onClick: () => void; children: React.ReactNode; destructive?: boolean }) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <Button variant={props.destructive ? "destructive" : "ghost"} size="icon-sm" aria-label={props.label} onClick={props.onClick} />
        }
      >
        {props.children}
      </TooltipTrigger>
      <TooltipContent>{props.label}</TooltipContent>
    </Tooltip>
  )
}

function JobRow({ job }: { job: Job }) {
  const { t, i18n } = useTranslation()
  const status = STATUS[job.status] ?? STATUS.queued
  const act = useMutation({
    mutationFn: (fn: () => Promise<unknown>) => fn(),
    onSettled: () => queryClient.invalidateQueries({ queryKey: keys.jobs }),
    onError: (error) => toast.error(error.message),
  })
  const created = new Date(job.created_at)

  return (
    <div className="flex items-center gap-3 rounded-lg border bg-card px-3 py-2.5">
      {job.outputs[0] ? (
        <img src={api.thumbUrl(job.outputs[0].id, 96)} alt="" className="size-10 shrink-0 rounded-md bg-checker object-cover" />
      ) : (
        <div className="grid size-10 shrink-0 place-items-center rounded-md bg-muted text-muted-foreground">
          {job.status === "running" ? <Loader2 className="size-4 animate-spin" aria-hidden /> : <ListOrdered className="size-4" aria-hidden />}
        </div>
      )}
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm">{job.summary || t("Untitled job")}</p>
        <p className="text-xs text-muted-foreground tabular-nums">
          {job.tab} · {Number.isNaN(created.getTime()) ? job.created_at : created.toLocaleTimeString(i18n.language)}
          {job.error ? <span className="text-destructive"> · {job.error}</span> : null}
        </p>
      </div>
      <Badge variant={status.variant}>{t(status.label)}</Badge>
      <div className="flex shrink-0 gap-0.5">
        {job.status === "running" ? (
          <JobAction label={t("Interrupt")} destructive onClick={() => act.mutate(() => api.jobs.interrupt(job.id))}>
            <Ban />
          </JobAction>
        ) : null}
        {job.status === "failed" || job.status === "interrupted" || job.status === "cancelled" ? (
          <JobAction label={t("Retry")} onClick={() => act.mutate(async () => {
                // A retried job is ours: its result shows on its form's page.
                const { job_id } = await api.jobs.retry(job.id)
                useProgress.getState().track(job_id, job.tab)
              })}>
            <RotateCw />
          </JobAction>
        ) : null}
        {job.status !== "running" ? (
          <JobAction label={job.status === "queued" ? t("Cancel") : t("Remove")} onClick={() => act.mutate(() => api.jobs.remove(job.id))}>
            <Trash2 />
          </JobAction>
        ) : null}
      </div>
    </div>
  )
}

const ROW_ESTIMATE = 68

export default function QueuePage() {
  // TanStack Virtual's virtualizer is a mutable object the React Compiler must not memoise.
  "use no memo"
  const { t } = useTranslation()
  const jobs = useJobs()
  const scroller = useRef<HTMLDivElement>(null)
  const items = jobs.data?.pages.flatMap((page) => page.items) ?? []
  const total = jobs.data?.pages[0]?.total ?? undefined
  // oxlint-disable-next-line react/incompatible-library -- opted out of memoisation above
  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => scroller.current,
    estimateSize: () => ROW_ESTIMATE,
    overscan: 8,
    getItemKey: (index) => items[index]?.id ?? index,
  })
  const virtualItems = virtualizer.getVirtualItems()
  const lastIndex = virtualItems.at(-1)?.index ?? -1
  const { hasNextPage, isFetchingNextPage, fetchNextPage } = jobs

  // Load the next page when the end of the list scrolls into view.
  useEffect(() => {
    if (lastIndex >= items.length - 1 && hasNextPage && !isFetchingNextPage) void fetchNextPage()
  }, [lastIndex, items.length, hasNextPage, isFetchingNextPage, fetchNextPage])

  if (jobs.isPending) {
    return (
      <div className="space-y-2 p-4" aria-busy="true">
        {Array.from({ length: 4 }, (_, i) => (
          <Skeleton key={i} className="h-14 w-full" />
        ))}
      </div>
    )
  }
  if (jobs.isError) {
    return <PagePlaceholder icon={ListOrdered} title="Queue unavailable" description={jobs.error.message} />
  }
  if (items.length === 0) {
    return <PagePlaceholder icon={ListOrdered} title="The queue is empty" description="Jobs you generate show up here, with their history." />
  }
  return (
    <div ref={scroller} className="h-full overflow-y-auto">
      <div className="mx-auto max-w-4xl p-4">
        <div role="list" aria-label={t("Jobs")} className="relative" style={{ height: virtualizer.getTotalSize() }}>
          {virtualItems.map((row) => {
            const job = items[row.index]
            if (!job) return null
            return (
              <div
                key={row.key}
                ref={virtualizer.measureElement}
                data-index={row.index}
                role="listitem"
                aria-setsize={total ?? (hasNextPage ? -1 : items.length)}
                aria-posinset={row.index + 1}
                className="absolute inset-x-0 top-0 pb-2"
                style={{ transform: `translateY(${row.start}px)` }}
              >
                <JobRow job={job} />
              </div>
            )
          })}
        </div>
        {hasNextPage ? (
          <div className="flex justify-center pt-2">
            <Button variant="outline" size="sm" onClick={() => void fetchNextPage()} disabled={isFetchingNextPage}>
              {isFetchingNextPage ? <Loader2 className="animate-spin" aria-hidden /> : null}
              {isFetchingNextPage ? t("Loading…") : t("Load more")}
            </Button>
          </div>
        ) : null}
      </div>
    </div>
  )
}
