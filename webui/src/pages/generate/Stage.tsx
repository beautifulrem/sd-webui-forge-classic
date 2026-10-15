/**
 * The stage: the selected result large, or the live preview while one of
 * this client's generation jobs runs (unless the user picked an older
 * result), with step / percent / ETA over the image, the result actions and
 * the generation parameters.
 */
import { ChevronDown, Expand, ImageIcon, Radio } from "lucide-react"
import { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { useClientOption } from "@/api/app-options"
import { api } from "@/api/endpoints"
import { progressFraction, type RunningJob } from "@/api/progress"
import { useWs } from "@/api/ws"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils"
import { selectedItem, useSession } from "@/features/gallery/session-store"
import { InfotextPanel } from "@/features/infotext/InfotextPanel"
import { formatEta } from "@/features/progress/effects"
import type { GenerateTab } from "@/features/progress/run-controller"
import { ResultActions } from "@/features/send-to/ResultActions"
import { useOwnRunningJob } from "./hooks"

function ProgressOverlay({ job }: { job: RunningJob }) {
  const { t } = useTranslation()
  const fraction = progressFraction(job) ?? 0
  const eta = formatEta(job.eta)
  return (
    <div className="absolute inset-x-3 bottom-3 rounded-lg bg-background/80 p-2.5 shadow-sm backdrop-blur">
      <div className="mb-1.5 flex items-center justify-between gap-2 text-xs tabular-nums">
        <span className="truncate">
          {job.text ? t(job.text) : t("Step {{step}} of {{steps}}", { step: job.step, steps: job.steps })}
          {job.jobCount && job.jobCount > 1 ? ` · ${t("Batch {{index}} of {{count}}", { index: (job.jobNo ?? 0) + 1, count: job.jobCount })}` : ""}
        </span>
        <span className="shrink-0 text-muted-foreground">
          {Math.round(fraction * 100)}%{eta ? ` · ${t("ETA {{eta}}", { eta })}` : ""}
        </span>
      </div>
      <Progress value={fraction * 100} aria-label={t("Generation progress")} />
    </div>
  )
}

/** Announces start and end of generation to screen readers (not every step). */
function LiveAnnouncer({ running }: { running: boolean }) {
  const { t } = useTranslation()
  const [message, setMessage] = useState("")
  const [was, setWas] = useState(running)
  if (was !== running) {
    setWas(running)
    setMessage(running ? t("Generation started") : t("Generation finished"))
  }
  return (
    <span role="status" className="sr-only">
      {message}
    </span>
  )
}

type StageProps = {
  tab: GenerateTab
  onOpenViewer: () => void
  className?: string
}

export function Stage({ tab, onOpenViewer, className }: StageProps) {
  const { t } = useTranslation()
  const job = useOwnRunningJob()
  const follow = useSession((s) => s.follow)
  const item = useSession(selectedItem)
  const showProgress = useClientOption("show_progressbar")
  const livePreviews = useClientOption("live_previews_enable")
  const preview = useWs((s) => (job && s.preview && (job.seq === undefined || s.preview.jobSeq === job.seq) ? s.preview.url : null))
  const live = job && follow && livePreviews ? preview : null
  const src = live ?? (item ? api.fileUrl(item.fileId) : null)
  const [infoOpen, setInfoOpen] = useState(true)

  // A new job takes the stage again.
  const jobId = job?.jobId ?? null
  useEffect(() => {
    if (jobId) useSession.getState().setFollow(true)
  }, [jobId])

  return (
    <section aria-label={t("Result")} className={cn("flex min-h-0 flex-col gap-2", className)}>
      <div className="relative flex min-h-64 flex-1 items-center justify-center overflow-hidden rounded-xl border bg-canvas">
        {src ? (
          <button
            type="button"
            onClick={onOpenViewer}
            aria-label={t("Open image viewer")}
            className="group/stage flex size-full items-center justify-center outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
          >
            <img
              src={src}
              alt={live ? t("Live preview") : t("Generated image")}
              draggable={!live}
              // Previews are often latent-sized (e.g. 96 px): scale them up to the stage.
              className={cn(live ? "size-full object-contain" : "max-h-full max-w-full object-contain", job && !live && "opacity-90")}
            />
          </button>
        ) : (
          <div className="flex flex-col items-center gap-2 p-6 text-center text-sm text-muted-foreground">
            <ImageIcon className="size-8 opacity-50" aria-hidden />
            <p>{job ? t("Waiting for the first preview…") : t("Generated images appear here")}</p>
          </div>
        )}
        {job && !follow ? (
          <Button
            variant="secondary"
            size="sm"
            onClick={() => useSession.getState().setFollow(true)}
            className="absolute top-3 left-1/2 -translate-x-1/2 gap-1.5 shadow-sm"
          >
            <Radio className="text-primary" />
            {t("Show live preview")}
          </Button>
        ) : null}
        {job && showProgress ? <ProgressOverlay job={job} /> : null}
        <LiveAnnouncer running={job !== null} />
      </div>

      <div className="flex flex-wrap items-center gap-1">
        <ResultActions item={live ? null : item} tab={tab} />
        <Button variant="ghost" size="icon-sm" className="ml-auto" aria-label={t("Open image viewer")} onClick={onOpenViewer} disabled={!src}>
          <Expand />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          aria-expanded={infoOpen}
          onClick={() => setInfoOpen((open) => !open)}
          className="gap-1 text-muted-foreground"
        >
          {t("Parameters")}
          <ChevronDown className={cn("transition-transform", !infoOpen && "-rotate-90")} />
        </Button>
      </div>
      {infoOpen && item && !live ? <InfotextPanel infotext={item.infotext} className="max-h-48 overflow-y-auto rounded-lg border bg-card p-3" /> : null}
    </section>
  )
}
