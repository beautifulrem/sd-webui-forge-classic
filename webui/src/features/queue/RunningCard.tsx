/** The job the worker runs now: live preview, progress, and interrupt / skip. */
import { useQuery } from "@tanstack/react-query"
import { Ban, Hourglass, Pause, SkipForward, Sparkles } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useClientOption } from "@/api/app-options"
import { api } from "@/api/endpoints"
import { progressFraction, useProgress } from "@/api/progress"
import { useWs } from "@/api/ws"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { formatEta, stepText } from "@/features/progress/effects"
import { act, formLabel, jobTitle } from "./queries"

export function RunningCard({ paused, queued }: { paused: boolean; queued: number }) {
  const { t } = useTranslation()
  const running = useProgress((s) => s.running)
  const livePreviews = useClientOption("live_previews_enable")
  const preview = useWs((s) => (running && s.preview && (running.seq === undefined || s.preview.jobSeq === running.seq) ? s.preview.url : null))
  const detail = useQuery({
    queryKey: ["jobs", "detail", running?.jobId],
    queryFn: ({ signal }) => api.jobs.get(running!.jobId, signal),
    enabled: running !== null,
    staleTime: 60_000,
  })

  if (!running) {
    return (
      <section aria-label={t("Now running")} className="flex items-center gap-3 rounded-xl border border-dashed bg-card/50 px-4 py-3">
        <span className="grid size-10 shrink-0 place-items-center rounded-lg bg-muted text-muted-foreground">
          {paused ? <Pause className="size-4" aria-hidden /> : <Sparkles className="size-4" aria-hidden />}
        </span>
        <div className="min-w-0">
          <p className="text-sm font-medium">{paused ? t("The queue is paused") : t("Nothing is running")}</p>
          <p className="text-xs text-muted-foreground">
            {paused
              ? queued > 0
                ? t("{{count}} jobs wait until you resume it.", { count: queued })
                : t("Jobs you add wait until you resume it.")
              : t("Queued jobs start here, one at a time.")}
          </p>
        </div>
      </section>
    )
  }

  const fraction = progressFraction(running) ?? 0
  const eta = formatEta(running.eta)
  const job = detail.data
  return (
    <section aria-label={t("Now running")} className="overflow-hidden rounded-xl border bg-card shadow-sm">
      <div className="flex gap-3 p-3">
        <div className="relative grid size-24 shrink-0 place-items-center overflow-hidden rounded-lg bg-checker sm:size-28">
          {preview && livePreviews ? (
            <img src={preview} alt={t("Live preview")} className="size-full object-contain" />
          ) : (
            // static: no endless animation while a job runs (see NavRail)
            <Hourglass className="size-5 text-muted-foreground" aria-hidden />
          )}
        </div>
        <div className="flex min-w-0 flex-1 flex-col gap-1.5">
          <div className="flex items-center gap-1.5">
            <Badge>{t("Running")}</Badge>
            {job ? (
              <span className="text-xs text-muted-foreground">
                #{job.seq} · {formLabel(job.tab)}
              </span>
            ) : null}
          </div>
          <p className="line-clamp-2 text-sm break-words">{job ? jobTitle(job) : t("Untitled job")}</p>
          <div className="mt-auto flex flex-col gap-1">
            <Progress value={fraction * 100} aria-label={t("Generation progress")} className="h-1.5" />
            <p className="flex flex-wrap gap-x-2 text-xs text-muted-foreground tabular-nums">
              <span>{stepText(running)}</span>
              {running.jobCount && running.jobCount > 1 ? <span>{t("Batch {{index}} of {{count}}", { index: (running.jobNo ?? 0) + 1, count: running.jobCount })}</span> : null}
              <span>{Math.round(fraction * 100)}%</span>
              {eta ? <span>{t("ETA {{eta}}", { eta })}</span> : null}
            </p>
          </div>
        </div>
        <div className="flex shrink-0 flex-col gap-1">
          <Button variant="outline" size="icon-sm" aria-label={t("Skip")} title={t("Skip")} onClick={() => void act(() => api.jobs.skip(running.jobId), t("Could not skip"))}>
            <SkipForward />
          </Button>
          <Button variant="destructive" size="icon-sm" aria-label={t("Interrupt")} title={t("Interrupt")} onClick={() => void act(() => api.jobs.interrupt(running.jobId), t("Could not interrupt"))}>
            <Ban />
          </Button>
        </div>
      </div>
    </section>
  )
}
