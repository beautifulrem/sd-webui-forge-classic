/**
 * Finished jobs, newest first, paged and virtualized: status, duration,
 * output thumbnails (open the viewer), star, queue again, open in its form,
 * remove. Imported jobs that cannot run as they are say why.
 */
import { useVirtualizer } from "@tanstack/react-virtual"
import { AlertTriangle, CheckCircle2, CircleSlash, FileInput, ImageOff, Loader2, RotateCw, Star, Trash2, XCircle } from "lucide-react"
import { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { api } from "@/api/endpoints"
import type { Job, JobStatus } from "@/api/types"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { openJob } from "./edit-session"
import { noteText } from "./notes"
import { durationOf, formatDuration, formLabel, removeJob, requeue, starJob } from "./queries"

const STATUS: Record<JobStatus, { label: string; icon: typeof CheckCircle2; tone: string }> = {
  completed: { label: "Completed", icon: CheckCircle2, tone: "text-success" },
  failed: { label: "Failed", icon: XCircle, tone: "text-destructive" },
  interrupted: { label: "Interrupted", icon: CircleSlash, tone: "text-warning" },
  cancelled: { label: "Cancelled", icon: CircleSlash, tone: "text-muted-foreground" },
  queued: { label: "Queued", icon: Loader2, tone: "text-muted-foreground" },
  running: { label: "Running", icon: Loader2, tone: "text-primary" },
}

const THUMBS = 4

function IconAction(props: { label: string; onClick: () => void; children: React.ReactNode; disabled?: boolean; pressed?: boolean; className?: string }) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={<Button variant="ghost" size="icon-sm" aria-label={props.label} aria-pressed={props.pressed} disabled={props.disabled} onClick={props.onClick} className={props.className} />}
      >
        {props.children}
      </TooltipTrigger>
      <TooltipContent>{props.label}</TooltipContent>
    </Tooltip>
  )
}

/** An output thumbnail; an image deleted since (e.g. from the gallery) shows a placeholder, not a broken image. */
function OutputThumb({ id }: { id: string }) {
  const { t } = useTranslation()
  const [missing, setMissing] = useState(false)
  if (missing)
    return (
      <span className="grid size-full place-items-center bg-muted text-muted-foreground" title={t("The image is no longer available")}>
        <ImageOff className="size-4" aria-hidden />
      </span>
    )
  return <img src={api.thumbUrl(id, 128)} alt="" loading="lazy" className="size-full object-cover" onError={() => setMissing(true)} />
}

function HistoryRow({ job, onView }: { job: Job; onView: (job: Job, index: number) => void }) {
  const { t, i18n } = useTranslation()
  const status = STATUS[job.status] ?? STATUS.completed
  const Icon = status.icon
  const images = job.outputs.filter((output) => (output.kind ?? "image") === "image")
  const created = new Date(job.created_at)
  const duration = durationOf(job)
  const [notesOpen, setNotesOpen] = useState(false)
  return (
    <div className="flex gap-3 rounded-lg border bg-card p-2.5">
      <div className="flex shrink-0 gap-1">
        {images.length > 0 ? (
          images.slice(0, THUMBS).map((output, index) => (
            <button
              key={output.id}
              type="button"
              onClick={() => onView(job, index)}
              aria-label={t("View image {{index}} of job #{{seq}}", { index: index + 1, seq: job.seq })}
              className={cn("relative size-14 overflow-hidden rounded-md bg-checker outline-none focus-visible:ring-3 focus-visible:ring-ring", index > 0 && "hidden sm:block")}
            >
              <OutputThumb id={output.id} />
              {index === THUMBS - 1 && images.length > THUMBS ? (
                <span className="absolute inset-0 grid place-items-center bg-black/55 text-xs font-semibold text-white">+{images.length - THUMBS + 1}</span>
              ) : null}
            </button>
          ))
        ) : (
          <span className={cn("grid size-14 place-items-center rounded-md bg-muted", status.tone)}>
            <Icon className="size-5" aria-hidden />
          </span>
        )}
      </div>
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <div className="flex items-start gap-2">
          <p className="min-w-0 flex-1 truncate text-sm" title={job.summary ?? undefined}>
            {job.name || job.summary || t("Untitled job")}
          </p>
          <span className={cn("inline-flex shrink-0 items-center gap-1 text-xs font-medium", status.tone)}>
            <Icon className="size-3.5" aria-hidden />
            {t(status.label)}
          </span>
        </div>
        <p className="flex flex-wrap items-center gap-x-1.5 text-xs text-muted-foreground tabular-nums">
          <span>#{job.seq}</span>
          <span>{formLabel(job.tab)}</span>
          <span>{Number.isNaN(created.getTime()) ? job.created_at : created.toLocaleString(i18n.language, { dateStyle: "short", timeStyle: "short" })}</span>
          {duration !== null ? <span>· {formatDuration(duration)}</span> : null}
          {images.length > 1 ? <span>· {t("{{count}} images", { count: images.length })}</span> : null}
          {job.source ? (
            <Badge variant="outline" className="h-4 px-1 text-[0.65rem]">
              {t("Imported")}
            </Badge>
          ) : null}
        </p>
        {job.error ? <p className="line-clamp-2 text-xs break-words text-destructive">{job.error}</p> : null}
        {!job.replayable || job.notes.length > 0 ? (
          <div className="text-xs">
            <button type="button" className="inline-flex items-center gap-1 text-warning hover:underline" aria-expanded={notesOpen} onClick={() => setNotesOpen((open) => !open)}>
              <AlertTriangle className="size-3.5" aria-hidden />
              {job.replayable ? t("Import notes") : t("Not replayable as it is: open it in its form to queue it")}
            </button>
            {notesOpen ? (
              <ul className="mt-1 list-disc pl-5 text-muted-foreground">
                {job.notes.map((note, index) => (
                  <li key={index}>{noteText(note)}</li>
                ))}
              </ul>
            ) : null}
          </div>
        ) : null}
      </div>
      <div className="flex shrink-0 flex-col items-end justify-between gap-0.5 sm:flex-row sm:items-start">
        <IconAction label={job.starred ? t("Unstar") : t("Star")} pressed={job.starred} onClick={() => void starJob(job, !job.starred)}>
          <Star className={cn(job.starred && "fill-amber-400 text-amber-400")} />
        </IconAction>
        <IconAction label={t("Queue again")} disabled={!job.replayable} onClick={() => void requeue(job)}>
          <RotateCw />
        </IconAction>
        <IconAction label={t("Open in its form")} onClick={() => void openJob(job.id)}>
          <FileInput />
        </IconAction>
        <IconAction label={t("Remove from history")} className="hover:text-destructive" onClick={() => void removeJob(job)}>
          <Trash2 />
        </IconAction>
      </div>
    </div>
  )
}

type HistoryListProps = {
  jobs: readonly Job[]
  total: number | null
  hasMore: boolean
  loadingMore: boolean
  onLoadMore: () => void
  onView: (job: Job, index: number) => void
}

export function HistoryList({ jobs, total, hasMore, loadingMore, onLoadMore, onView }: HistoryListProps) {
  // TanStack Virtual's virtualizer is a mutable object the React Compiler must not memoise.
  "use no memo"
  const { t } = useTranslation()
  const [scroller, setScroller] = useState<HTMLDivElement | null>(null)
  // oxlint-disable-next-line react/incompatible-library -- opted out of memoisation above
  const virtualizer = useVirtualizer({
    count: jobs.length,
    getScrollElement: () => scroller,
    estimateSize: () => 84,
    overscan: 6,
    getItemKey: (index) => jobs[index]?.id ?? index,
  })
  const items = virtualizer.getVirtualItems()
  const last = items.at(-1)?.index ?? -1
  useEffect(() => {
    if (last >= jobs.length - 3 && hasMore && !loadingMore) onLoadMore()
  }, [last, jobs.length, hasMore, loadingMore, onLoadMore])

  return (
    <div ref={setScroller} className="min-h-0 flex-1 overflow-y-auto">
      <div role="list" aria-label={t("Job history")} className="relative" style={{ height: virtualizer.getTotalSize() }}>
        {items.map((row) => {
          const job = jobs[row.index]
          if (!job) return null
          return (
            <div
              key={row.key}
              ref={virtualizer.measureElement}
              data-index={row.index}
              role="listitem"
              aria-posinset={row.index + 1}
              aria-setsize={total ?? (hasMore ? -1 : jobs.length)}
              className="absolute inset-x-0 top-0 pb-1.5"
              style={{ transform: `translateY(${row.start}px)` }}
            >
              <HistoryRow job={job} onView={onView} />
            </div>
          )
        })}
      </div>
      {loadingMore ? <p className="py-2 text-center text-xs text-muted-foreground">{t("Loading…")}</p> : null}
    </div>
  )
}
