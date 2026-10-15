/**
 * `task-runner`: a button that starts a page's long-running task and follows
 * it: progress bar (indeterminate without a fraction), the task's status
 * text, Cancel. The action answers at once with the task's status
 * (`remi.pages.Page.start_task`); the widget then polls `__task__` until the
 * task ends, whose answer carries the task's result patch (outputs, values,
 * a toast). Coming back to the page shows a task still running.
 *
 * props: { action: string, task?: string (default: action), label: string,
 *          variant?: "primary" | "secondary" (default "primary"), interval?: number (ms, 750) }
 * value: unused (null)
 */
import { Loader2, Play, X } from "lucide-react"
import { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import type { ActionExtra } from "@/api/types"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { useForm, useFormState, useOwner, useRunAction } from "@/ui/hooks"
import { messageText } from "@/ui/messages"
import type { WidgetProps } from "@/ui/widgets"
import { registerWidgetCatalogs } from "../_lib/i18n"
import { followTask, isFollowing, isTaskStatus, outputKey, resumeTask } from "./follow"

const NS = registerWidgetCatalogs("task-runner", import.meta.glob("./locales/*.json", { eager: true, import: "default" }))

type Props = { action: string; task?: string; label?: string; variant?: "primary" | "secondary"; interval?: number }

export default function TaskRunner({ props, disabled, id }: WidgetProps<null, Props>) {
  const { t: tw, i18n } = useTranslation(NS)
  const { store, formId } = useForm()
  const owner = useOwner()
  const run = useRunAction()
  const task = props.task ?? props.action
  const raw = useFormState((s) => s.outputs[owner]?.[outputKey(task)])
  const status = isTaskStatus(raw) ? raw : null
  const [starting, setStarting] = useState(false)
  const active = starting || (status !== null && !status.done)
  const ctx = { store, tab: formId }

  // A task started earlier (another visit, a reload): show it and follow it.
  useEffect(() => {
    let cancelled = false
    const context = { store, tab: formId }
    if (isFollowing(context, owner, task)) return
    void resumeTask(context, owner, task).then((current) => {
      if (!cancelled && current && !current.done) void followTask(context, owner, task, current.id, props.interval)
    })
    return () => {
      cancelled = true
    }
  }, [store, formId, owner, task, props.interval])

  const start = async () => {
    setStarting(true)
    try {
      const answer = await run(props.action)
      const started = answer?.outputs?.[outputKey(task)]
      if (isTaskStatus(started) && !started.done) void followTask(ctx, owner, task, started.id, props.interval)
    } finally {
      setStarting(false)
    }
  }

  const cancel = () => void run("__cancel__", { task } as unknown as ActionExtra)

  const percent = status?.progress != null ? Math.round(status.progress * 100) : null
  const text = status?.text ? messageText(i18n, status.text) : status?.state === "queued" ? tw("Queued") : status?.state === "running" ? tw("Running…") : ""
  const label = props.label ? tw(props.label) : tw("Run")
  // Screen readers hear state changes only (the bar carries the percentage);
  // a failure is announced by its alert.
  const announced =
    status?.state === "queued"
      ? tw("Queued")
      : status?.state === "running"
        ? tw("Running…")
        : status?.state === "completed"
          ? tw("Finished")
          : status?.state === "cancelled"
            ? tw("Cancelled")
            : ""

  return (
    <div className="flex flex-col gap-2">
      <span className="sr-only" role="status" aria-live="polite" aria-atomic>
        {announced ? `${label}: ${announced}` : ""}
      </span>
      <div className="flex flex-wrap items-center gap-2">
        <Button
          id={id}
          variant={props.variant === "secondary" ? "secondary" : "default"}
          disabled={disabled || active}
          aria-busy={active || undefined}
          onClick={() => void start()}
        >
          {active ? <Loader2 className="animate-spin" aria-hidden /> : <Play aria-hidden />}
          {label}
        </Button>
        {active && status ? (
          <Button variant="ghost" size="sm" disabled={status.cancelling} onClick={cancel}>
            <X aria-hidden />
            {status.cancelling ? tw("Cancelling…") : tw("Cancel")}
          </Button>
        ) : null}
      </div>
      {active && status ? (
        <div className="flex flex-col gap-1">
          {/* Base UI: a null value is an indeterminate bar */}
          <Progress value={percent} aria-label={label} />
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="min-w-0 flex-1 truncate">{text}</span>
            {percent != null ? <span className="tabular">{percent}%</span> : null}
          </div>
        </div>
      ) : status?.state === "failed" && status.error ? (
        <p className="text-xs text-destructive" role="alert">
          {tw("Failed: {{error}}", { error: status.error })}
        </p>
      ) : null}
    </div>
  )
}
