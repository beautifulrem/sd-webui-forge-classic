/** Confirm dialog and reconnecting overlay for "Reload UI" / "Restart server". */
import { AlertTriangle, Check, Loader2, Power, RefreshCw } from "lucide-react"
import { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { useProgress } from "@/api/progress"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogMedia,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import {
  cancelServerCommand,
  dismissServerCommand,
  keepWaiting,
  runServerCommand,
  type ServerCommand,
  type ServerPhase,
  useServerControl,
} from "./server-control"
import { leaveWithoutAsking, unsavedTotal, useUnsaved } from "./unsaved"

const TITLES: Record<ServerCommand, string> = { reload: "Reload UI", restart: "Restart server" }

function ConfirmDialog({ command }: { command: ServerCommand }) {
  const { t } = useTranslation()
  const unsaved = useUnsaved((s) => unsavedTotal(s.sources))
  const running = useProgress((s) => s.running !== null)
  const Icon = command === "reload" ? RefreshCw : Power
  return (
    <AlertDialog open onOpenChange={(open) => !open && cancelServerCommand()}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogMedia>
            <Icon />
          </AlertDialogMedia>
          <AlertDialogTitle>{t(TITLES[command])}</AlertDialogTitle>
          <AlertDialogDescription>
            {command === "reload"
              ? t("Reloads scripts and extensions and rebuilds the app. Queued jobs wait and continue afterwards.")
              : t("Stops the WebUI process and starts it again. The running job is interrupted; queued jobs continue after the restart.")}
          </AlertDialogDescription>
        </AlertDialogHeader>
        {running && command === "reload" ? (
          <p className="rounded-lg bg-muted px-3 py-2 text-sm text-muted-foreground">{t("A job is running: the reload starts when it has finished.")}</p>
        ) : null}
        {unsaved > 0 ? (
          <p role="alert" className="flex items-start gap-2 rounded-lg bg-warning/15 px-3 py-2 text-sm">
            <AlertTriangle aria-hidden className="mt-0.5 size-4 shrink-0 text-warning" />
            {t("{{count}} unsaved settings changes will be lost.", { count: unsaved })}
          </p>
        ) : null}
        <AlertDialogFooter>
          <AlertDialogCancel>{t("Cancel")}</AlertDialogCancel>
          <AlertDialogAction
            onClick={() => void runServerCommand(command)}
          >
            {t(TITLES[command])}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}

const STEPS: { phase: ServerPhase; label: string }[] = [
  { phase: "waiting-down", label: "Stopping the server" },
  { phase: "waiting-up", label: "Starting again" },
  { phase: "done", label: "Reloading this page" },
]
const ORDER: ServerPhase[] = ["requesting", "waiting-down", "waiting-up", "done"]

function useElapsed(since: number, running: boolean): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!running) return
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [running])
  return Math.max(0, Math.round((now - since) / 1000))
}

function ReconnectOverlay() {
  const { t } = useTranslation()
  const { command, phase, startedAt, error, canWait } = useServerControl()
  const jobRunning = useProgress((s) => s.running !== null)
  const failed = phase === "failed"
  const elapsed = useElapsed(startedAt, !failed)
  const current = ORDER.indexOf(phase)
  const title = command === "restart" ? t("Restarting the server…") : t("Reloading the UI…")

  return (
    <AlertDialog open>
      <AlertDialogContent className="sm:max-w-sm" aria-busy={!failed}>
        <AlertDialogHeader>
          <AlertDialogMedia className={cn(failed && "bg-destructive/10 text-destructive")}>
            {failed ? <AlertTriangle /> : <Loader2 className="animate-spin" />}
          </AlertDialogMedia>
          <AlertDialogTitle>{failed ? t("The server is not back") : title}</AlertDialogTitle>
          <AlertDialogDescription aria-live="polite">
            {failed && error
              ? t(error)
              : phase === "waiting-down" && jobRunning && command === "reload"
                ? t("Waiting for the running job to finish…")
                : t("This page reloads by itself once the server answers again.")}
          </AlertDialogDescription>
        </AlertDialogHeader>
        {!failed ? (
          <ol className="flex flex-col gap-2 text-sm" aria-label={t("Progress")}>
            {STEPS.map((step) => {
              const index = ORDER.indexOf(step.phase)
              const done = current > index
              const active = current === index
              return (
                <li key={step.phase} className={cn("flex items-center gap-2", !done && !active && "text-muted-foreground")} aria-current={active ? "step" : undefined}>
                  <span className={cn("grid size-5 place-items-center rounded-full border", done && "border-success bg-success text-background", active && "border-primary")}>
                    {done ? <Check className="size-3" aria-hidden /> : active ? <span className="size-1.5 animate-pulse rounded-full bg-primary" /> : null}
                  </span>
                  {t(step.label)}
                </li>
              )
            })}
          </ol>
        ) : null}
        <p className="text-xs text-muted-foreground tabular-nums">{t("{{seconds}} s elapsed", { seconds: elapsed })}</p>
        <AlertDialogFooter>
          {failed ? (
            <>
              <Button variant="outline" onClick={dismissServerCommand}>
                {t("Close")}
              </Button>
              {canWait ? (
                <Button variant="outline" onClick={() => keepWaiting()}>
                  {t("Keep waiting")}
                </Button>
              ) : null}
            </>
          ) : null}
          <Button
            variant={failed ? "default" : "outline"}
            onClick={() => {
              leaveWithoutAsking()
              location.reload()
            }}
          >
            {t("Reload page now")}
          </Button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}

export default function ServerOverlay() {
  const pending = useServerControl((s) => s.pending)
  const phase = useServerControl((s) => s.phase)
  if (phase !== "idle") return <ReconnectOverlay />
  return pending ? <ConfirmDialog command={pending} /> : null
}
