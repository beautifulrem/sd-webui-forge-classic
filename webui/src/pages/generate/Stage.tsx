/**
 * The stage: the selected result large, or the live preview while one of
 * this client's generation jobs runs (unless the user picked an older
 * result), with step / percent / ETA over the image, the result actions and
 * the generation parameters.
 */
import { ChevronDown, Expand, Radio } from "@/components/icons"
import * as m from "motion/react-m"
import { type ReactNode, useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { useClientOption } from "@/api/app-options"
import { api } from "@/api/endpoints"
import { progressFraction, type RunningJob } from "@/api/progress"
import { useWs } from "@/api/ws"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { selectedItem, useSession } from "@/features/gallery/session-store"
import { InfotextPanel } from "@/features/infotext/InfotextPanel"
import { formatShortcut } from "@/app/commands"
import { ExposureSheet } from "@/features/progress/ExposureSheet"
import { formatEta, stepText } from "@/features/progress/effects"
import type { GenerateTab } from "@/features/progress/run-controller"
import { ResultActions } from "@/features/send-to/ResultActions"
import { Img2imgStage } from "@/canvas"
import { useOwnRunningJob } from "./hooks"
import { LivePreview } from "./LivePreview"

function ProgressOverlay({ job }: { job: RunningJob }) {
  const { t } = useTranslation()
  const fraction = progressFraction(job) ?? 0
  const eta = formatEta(job.eta)
  return (
    // No backdrop-filter: the preview under it changes every step, and a blur would be redrawn with it.
    <div className="absolute inset-x-3 bottom-3 rounded-md bg-background/92 px-3 pt-2 pb-2.5">
      <div className="mb-1.5 flex items-baseline gap-3 text-xs tabular-nums">
        <span className="truncate">{stepText(job)}</span>
        {job.jobCount && job.jobCount > 1 ? (
          <span className="shrink-0 text-muted-foreground">{t("Batch {{index}} of {{count}}", { index: (job.jobNo ?? 0) + 1, count: job.jobCount })}</span>
        ) : null}
        <span className="ml-auto shrink-0 font-medium">{Math.round(fraction * 100)}%</span>
        {eta ? <span className="shrink-0 text-muted-foreground">{t("ETA {{eta}}", { eta })}</span> : null}
      </div>
      <ExposureSheet step={job.step} steps={job.steps} fraction={fraction} aria-label={t("Generation progress")} />
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
  /** Header items at the start / end (the pane toggles). */
  leading?: ReactNode
  trailing?: ReactNode
  /** A row above the header (the run controls while the parameters pane is collapsed). */
  toolbar?: ReactNode
}

/** Below this height the parameters panel starts collapsed: the image keeps the room. */
const COMPACT_STAGE_PX = 520

export function Stage({ tab, onOpenViewer, className, leading, trailing, toolbar }: StageProps) {
  const { t } = useTranslation()
  const job = useOwnRunningJob()
  const follow = useSession((s) => s.follow)
  const item = useSession(selectedItem)
  const showProgress = useClientOption("show_progressbar")
  const livePreviews = useClientOption("live_previews_enable")
  const videoAutoplay = useClientOption("video_player_auto")
  const videoLoop = useClientOption("video_player_loop")
  const preview = useWs((s) => (job && s.preview && (job.seq === undefined || s.preview.jobSeq === job.seq) ? s.preview.url : null))
  const live = job && follow && livePreviews ? preview : null
  const src = live ?? (item ? api.fileUrl(item.fileId) : null)
  const video = !live && item?.videoId ? item.videoId : null
  const [infoOpen, setInfoOpen] = useState(true)
  const section = useRef<HTMLElement>(null)

  // A new job takes the stage again.
  const jobId = job?.jobId ?? null
  useEffect(() => {
    if (jobId) useSession.getState().setFollow(true)
  }, [jobId])

  // A short stage (a small window, or one resized smaller) shows the image first; the parameters are
  // a click away. Only becoming short closes them: the user's choice stands until the next time.
  useEffect(() => {
    const element = section.current
    if (!element) return
    let short = false
    const check = () => {
      const height = element.clientHeight
      const now = height > 0 && height < COMPACT_STAGE_PX
      if (now && !short) setInfoOpen(false)
      short = now
    }
    check()
    if (typeof ResizeObserver === "undefined") return
    const observer = new ResizeObserver(check)
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  const image = (
    <div data-lit={job ? "" : undefined} className="relative flex min-h-48 flex-1 items-center justify-center overflow-hidden rounded-md bg-canvas data-lit:shadow-[inset_0_0_0_1px_color-mix(in_oklch,var(--glow)_30%,transparent),inset_0_0_80px_-20px_var(--glow-dim)]">
      {video && item ? (
        // A generated video (e.g. Wan): played here; its frames stay in the history and the viewer.
        <video
          key={video}
          src={api.fileUrl(video)}
          poster={api.fileUrl(item.fileId)}
          controls
          playsInline
          autoPlay={videoAutoplay}
          loop={videoLoop}
          aria-label={t("Generated video")}
          className={cn("max-h-full max-w-full object-contain", job && "opacity-90")}
        />
      ) : src ? (
        <button
          type="button"
          onClick={onOpenViewer}
          aria-label={t("Open image viewer")}
          className="group/stage flex size-full items-center justify-center outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
        >
          {live ? (
            <LivePreview src={live} alt={t("Live preview")} />
          ) : (
            // a finished picture is laid on the table: one short settle, only when the picture changes
            <m.img
              key={src}
              src={src}
              alt={t("Generated image")}
              initial={{ opacity: 0, scale: 0.985 }}
              animate={{ opacity: job ? 0.9 : 1, scale: 1 }}
              transition={{ duration: 0.2, ease: [0.2, 0, 0, 1] }}
              className="max-h-full max-w-full object-contain"
            />
          )}
        </button>
      ) : (
        <p className="max-w-[28ch] p-6 text-center text-sm text-balance text-muted-foreground">
          {job ? t("Waiting for the first preview…") : t("Write a prompt, then press {{shortcut}} to generate", { shortcut: formatShortcut("Mod+Enter") })}
        </p>
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
  )

  const actions = (
    <>
      <ResultActions item={live ? null : item} tab={tab} upscalable collapsible />
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
    </>
  )

  // Never more than about a quarter of the stage: the image keeps the room.
  const info = infoOpen && item && !live ? <InfotextPanel infotext={item.infotext} className="max-h-[max(4.5rem,min(12rem,25%))] shrink-0 overflow-y-auto rounded-sm px-1 py-1.5" /> : null
  const toolbarRow = toolbar ? (
    <div className="flex items-center gap-1">
      {leading}
      {toolbar}
      {trailing}
    </div>
  ) : null

  if (tab === "img2img") {
    return (
      <section ref={section} aria-label={t("Image area")} className={cn("flex min-h-0 flex-col gap-2", className)}>
        {toolbarRow}
        <Img2imgStage
          className="min-h-0 flex-1"
          leading={toolbar ? null : leading}
          trailing={toolbar ? null : trailing}
          actions={actions}
          result={
            <>
              {image}
              {info}
            </>
          }
        />
      </section>
    )
  }

  return (
    <section ref={section} aria-label={t("Result")} className={cn("flex min-h-0 flex-col gap-2", className)}>
      {toolbarRow}
      <div className="flex min-h-7 flex-wrap items-center gap-1">
        {toolbar ? null : leading}
        {actions}
        {toolbar ? null : trailing}
      </div>
      {image}
      {info}
    </section>
  )
}
