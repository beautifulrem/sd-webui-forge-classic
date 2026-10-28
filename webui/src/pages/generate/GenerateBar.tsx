/**
 * The sticky run controls of a form: Generate (Ctrl/Cmd+Enter; while a job
 * runs it queues another), Skip (Alt+Enter), Interrupt (Esc), a menu with
 * "generate forever", "generate N times" and notifications, and a status
 * line (queue position, step, ETA, forever / repeat state).
 */
import { useQuery } from "@tanstack/react-query"
import { Ban, Bell, ChevronDown, Infinity as InfinityIcon, Loader2, Repeat, RotateCcw, SkipForward, X } from "@/components/icons"
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { progressFraction, useProgress } from "@/api/progress"
import { queueItemsQuery } from "@/api/queries"
import { formatShortcut } from "@/app/commands"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Input } from "@/components/ui/input"
import { Kbd } from "@/components/ui/kbd"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { cn } from "@/lib/utils"
import { ExposureSheet } from "@/features/progress/ExposureSheet"
import { formatEta, stepText, notificationsAvailable } from "@/features/progress/effects"
import { dispatchRun, type GenerateTab, useRuns } from "@/features/progress/run-controller"
import { phaseOf } from "@/features/progress/run-machine"
import { EnqueueButton, QueueEditBanner } from "@/features/queue/FormQueueControls"
import { requestReset } from "./reset-request"
import { jobsAhead } from "./run-helpers"

function useQueuePosition(jobId: string | null): number | null {
  const queued = useQuery({ ...queueItemsQuery, enabled: jobId !== null })
  const running = useProgress((s) => s.running !== null)
  if (!jobId || !queued.data) return null
  return jobsAhead(queued.data, jobId, running)
}

function RepeatPopover({ tab, disabled }: { tab: GenerateTab; disabled?: boolean }) {
  const { t } = useTranslation()
  const [runs, setRuns] = useState(4)
  const [open, setOpen] = useState(false)
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger render={<Button variant="outline" size="icon-lg" className="size-11" aria-label={t("Generate several times")} disabled={disabled} />}>
        <Repeat />
      </PopoverTrigger>
      <PopoverContent aria-label={t("Generate several times")} side="top" align="end" className="w-60">
        <form
          className="flex flex-col gap-2"
          onSubmit={(event) => {
            event.preventDefault()
            dispatchRun(tab, { type: "generate", runs })
            setOpen(false)
          }}
        >
          <label htmlFor={`${tab}-repeat`} className="text-sm font-medium">
            {t("Generate several times")}
          </label>
          <p className="text-xs text-muted-foreground">{t("Each run starts when the previous one has finished and uses the form as it is then.")}</p>
          <div className="flex gap-1.5">
            <Input
              id={`${tab}-repeat`}
              type="number"
              min={2}
              max={999}
              value={runs}
              onChange={(event) => setRuns(Math.max(1, Math.min(999, Math.round(Number(event.target.value) || 1))))}
              className="h-8 tabular-nums"
            />
            <Button type="submit" size="sm">
              {t("Start", { context: "runs" })}
            </Button>
          </div>
        </form>
      </PopoverContent>
    </Popover>
  )
}

function StatusLine({ tab }: { tab: GenerateTab }) {
  const { t } = useTranslation()
  const run = useRuns((s) => s.runs[tab])
  const phase = phaseOf(run)
  const job = useProgress((s) => (s.running && run.jobs.includes(s.running.jobId) ? s.running : null))
  const firstQueued = phase === "queued" ? (run.jobs[0] ?? null) : null
  const ahead = useQueuePosition(firstQueued)
  const fraction = progressFraction(job)
  const eta = formatEta(job?.eta)

  // Each fact its own span, spaced apart: no "A · B · C" string.
  let parts: (string | null)[] = []
  if (phase === "submitting") parts = [t("Queueing…")]
  else if (phase === "interrupting") parts = [t("Interrupting…")]
  else if (phase === "queued") parts = [ahead !== null ? t("Queued, {{count}} ahead", { count: ahead }) : t("Queued")]
  else if (job) {
    parts = [
      stepText(job),
      job.jobCount && job.jobCount > 1 ? t("Batch {{index}} of {{count}}", { index: (job.jobNo ?? 0) + 1, count: job.jobCount }) : null,
      eta ? t("ETA {{eta}}", { eta }) : null,
    ]
  }
  const facts = parts.filter((part): part is string => Boolean(part))

  const extras = run.forever || run.repeatLeft > 0 || run.jobs.length > 1
  if (facts.length === 0 && !extras) return null
  return (
    <div className="flex flex-col gap-1.5">
      {phase === "running" && job && fraction !== null ? (
        <ExposureSheet step={job.step} steps={job.steps} fraction={fraction} aria-label={t("Generation progress")} />
      ) : null}
      <div className="flex min-h-5 flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
        {facts.map((fact, index) => (
          <span key={index} className={cn("tabular-nums", index === 0 && "text-foreground")}>
            {fact}
          </span>
        ))}
        {run.jobs.length > 1 ? <span className="tabular-nums">{t("{{count}} of yours queued", { count: run.jobs.length })}</span> : null}
        {run.forever ? (
          <span className="inline-flex items-center gap-1 rounded-sm bg-primary/12 py-0.5 pr-0.5 pl-1.5 text-primary">
            <InfinityIcon className="size-3.5" aria-hidden />
            {t("Generating forever")}
            <button
              type="button"
              aria-label={t("Stop generating forever")}
              onClick={() => dispatchRun(tab, { type: "forever", on: false })}
              className="grid size-4 place-items-center rounded-xs outline-none hover:bg-primary/20 focus-visible:ring-2 focus-visible:ring-ring"
            >
              <X className="size-3" />
            </button>
          </span>
        ) : null}
        {run.repeatLeft > 0 ? (
          <span className="inline-flex items-center gap-1 rounded-sm bg-muted py-0.5 pr-0.5 pl-1.5">
            <Repeat className="size-3.5" aria-hidden />
            {t("{{count}} more runs", { count: run.repeatLeft })}
            <button
              type="button"
              aria-label={t("Cancel the remaining runs")}
              onClick={() => dispatchRun(tab, { type: "stop-repeats" })}
              className="grid size-4 place-items-center rounded-xs outline-none hover:bg-foreground/10 focus-visible:ring-2 focus-visible:ring-ring"
            >
              <X className="size-3" />
            </button>
          </span>
        ) : null}
      </div>
    </div>
  )
}

export function GenerateBar({ tab, className }: { tab: GenerateTab; className?: string }) {
  const { t } = useTranslation()
  const run = useRuns((s) => s.runs[tab])
  const phase = phaseOf(run)
  const running = phase === "running" || phase === "interrupting"
  const busy = phase !== "idle"

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <QueueEditBanner tab={tab} />
      <div className="flex items-center gap-1.5">
        {/* a size container: the shortcut hint shows only when the label fits beside it */}
        <div className="@container/generate flex min-w-0 flex-1">
          <Button
            variant="glow"
            size="lg"
            className="h-11 min-w-0 flex-1 gap-2 rounded-r-none text-[0.95rem] font-semibold"
            onClick={() => dispatchRun(tab, { type: "generate" })}
            disabled={phase === "submitting"}
            aria-keyshortcuts="Control+Enter Meta+Enter"
          >
            {phase === "submitting" ? <Loader2 className="animate-spin" aria-hidden /> : null}
            <span className="truncate">{busy ? t("Queue another") : t("Generate")}</span>
            {busy ? null : (
              <Kbd className="ml-1 hidden border border-glow-foreground/30 bg-transparent font-normal text-glow-foreground/75 @min-[14rem]/generate:inline-flex touch-only:hidden!">
                {formatShortcut("Mod+Enter")}
              </Kbd>
            )}
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger
              render={
                <Button
                  variant="glow"
                  size="icon-lg"
                  className="h-11 w-9 rounded-l-none border-l border-glow-foreground/15"
                  aria-label={t("More generate options")}
                />
              }
            >
              <ChevronDown />
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="min-w-56">
              <DropdownMenuCheckboxItem checked={run.forever} onCheckedChange={(on) => dispatchRun(tab, { type: "forever", on })}>
                <InfinityIcon />
                {t("Generate forever")}
              </DropdownMenuCheckboxItem>
              {notificationsAvailable() ? (
                <DropdownMenuItem
                  onClick={() =>
                    void Notification.requestPermission().then((permission) =>
                      permission === "granted"
                        ? toast.success(t("Notifications are on"))
                        : toast.info(t("Notifications are blocked by the browser")),
                    )
                  }
                >
                  <Bell />
                  {t("Notify me when images are ready")}
                </DropdownMenuItem>
              ) : null}
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => requestReset(tab)}>
                <RotateCcw />
                {t("Reset all parameters")}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
        <RepeatPopover tab={tab} disabled={phase === "submitting"} />
        <EnqueueButton tab={tab} />
        {running ? (
          <>
            <Button
              variant="outline"
              size="icon-lg"
              className="size-11"
              onClick={() => dispatchRun(tab, { type: "skip" })}
              aria-label={t("Skip")}
              aria-keyshortcuts="Alt+Enter"
              title={`${t("Skip")} (${formatShortcut("Alt+Enter")})`}
              disabled={phase === "interrupting"}
            >
              <SkipForward />
            </Button>
          </>
        ) : null}
        {busy && phase !== "submitting" ? (
          <Button
            variant="destructive"
            size="icon-lg"
            className="size-11"
            onClick={() => dispatchRun(tab, { type: "interrupt" })}
            aria-label={t("Interrupt")}
            aria-keyshortcuts="Escape"
            title={`${t("Interrupt")} (Esc)`}
            disabled={phase === "interrupting"}
          >
            {phase === "interrupting" ? <Loader2 className="animate-spin" /> : <Ban />}
          </Button>
        ) : null}
      </div>
      <StatusLine tab={tab} />
    </div>
  )
}
