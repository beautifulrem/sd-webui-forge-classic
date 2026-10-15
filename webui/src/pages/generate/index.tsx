/**
 * txt2img / img2img: the schema-rendered form on the left, results on the
 * right (stacked on mobile). Generation submits the form as a job; progress
 * and results arrive over the WebSocket. Interrupting is a button or a
 * palette command, not a bare Escape (Escape closes popups and editors).
 */
import { useMutation } from "@tanstack/react-query"
import { Ban, Loader2, RotateCcw, SkipForward, Sparkles } from "lucide-react"
import { useEffect, useMemo } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { useDefaultLayout } from "react-resizable-panels"
import { api } from "@/api/endpoints"
import { validationErrors } from "@/api/http"
import { useProgress } from "@/api/progress"
import { useSchema } from "@/api/queries"
import { type Command, formatShortcut, useRegisterCommands } from "@/app/commands"
import type { Route } from "@/app/router"
import { Button } from "@/components/ui/button"
import { Kbd } from "@/components/ui/kbd"
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable"
import { useIsDesktop } from "@/lib/use-media-query"
import { resetParams } from "@/ui/actions"
import { getFormStore } from "@/ui/form-store"
import { TabForm } from "@/ui/SchemaForm"
import { jobRequest } from "@/ui/tab"
import type { TabSchema } from "@/ui/schema"
import { OutputPanel } from "./OutputPanel"

/** Submits the form as a job; a rejected request marks the offending fields (schema-format.md §3). */
function useSubmit(schema: TabSchema) {
  const { t } = useTranslation()
  const store = getFormStore(schema.tab)
  return useMutation({
    mutationFn: () => api.jobs.create(jobRequest(store, schema)),
    onSuccess: ({ job_id }) => {
      useProgress.getState().track(job_id, schema.tab)
      store.getState().clearErrors()
    },
    onError: (error) => {
      const errors = validationErrors(error)
      if (errors) store.getState().setErrors(errors)
      toast.error(errors ? t("Some parameters are invalid") : t("Could not queue the job"), { description: error.message })
    },
  })
}

/** Ctrl/Cmd+Enter generates (queues another job while one runs); held keys do not repeat. */
function useGenerateShortcut(generate: () => void) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (!(event.metaKey || event.ctrlKey) || event.key !== "Enter") return
      if (event.repeat || event.isComposing || event.defaultPrevented) return
      event.preventDefault()
      generate()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [generate])
}

function GenerateBar({ schema }: { schema: TabSchema }) {
  const { t } = useTranslation()
  const running = useProgress((s) => s.running)
  const store = getFormStore(schema.tab)
  const submit = useSubmit(schema)
  const generate = submit.mutate

  // The worker is shared: interrupt / skip act on whatever job is running.
  const interrupt = () => {
    const job = useProgress.getState().running
    if (job) void api.jobs.interrupt(job.jobId).catch(() => {})
  }
  const skip = () => {
    const job = useProgress.getState().running
    if (job) void api.jobs.skip(job.jobId).catch(() => {})
  }

  useGenerateShortcut(() => generate())

  const commands = useMemo<Command[]>(
    () => [
      { id: "generate", label: "Generate", group: "Generation", icon: Sparkles, shortcut: "Mod+Enter", run: () => generate() },
      { id: "interrupt", label: "Interrupt generation", group: "Generation", icon: Ban, run: interrupt },
      { id: "skip", label: "Skip current image", group: "Generation", icon: SkipForward, run: skip },
      {
        id: "reset-form",
        label: "Reset all parameters",
        group: "Generation",
        icon: RotateCcw,
        run: () => {
          const ctx = { store, tab: schema.tab }
          for (const owner of Object.keys(store.getState().defaults)) resetParams(ctx, owner)
        },
      },
    ],
    // `generate` (mutate) is stable; interrupt/skip read the store when run.
    // oxlint-disable-next-line react-hooks/exhaustive-deps
    [store, schema.tab],
  )
  useRegisterCommands(`generate:${schema.tab}`, commands)

  return (
    <div className="flex items-center gap-2">
      <Button
        size="lg"
        className="h-10 flex-1 gap-2 text-sm font-semibold shadow-sm"
        onClick={() => generate()}
        disabled={submit.isPending}
        aria-keyshortcuts="Control+Enter Meta+Enter"
      >
        {submit.isPending ? <Loader2 className="animate-spin" aria-hidden /> : <Sparkles aria-hidden />}
        {running ? t("Queue") : t("Generate")}
        <Kbd className="ml-1 hidden bg-primary-foreground/15 text-primary-foreground sm:inline-flex">{formatShortcut("Mod+Enter")}</Kbd>
      </Button>
      {running ? (
        <>
          <Button variant="outline" size="icon-lg" className="size-10" onClick={skip} aria-label={t("Skip")}>
            <SkipForward />
          </Button>
          <Button variant="destructive" size="icon-lg" className="size-10" onClick={interrupt} aria-label={t("Interrupt")}>
            <Ban />
          </Button>
        </>
      ) : null}
    </div>
  )
}

function FormColumn({ schema }: { schema: TabSchema }) {
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="sticky top-0 z-10 border-b bg-background/85 p-3 backdrop-blur md:static md:border-b-0 md:pb-0">
        <GenerateBar schema={schema} />
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain p-3">
        <TabForm schema={schema} />
      </div>
    </div>
  )
}

export default function GeneratePage({ route }: { route: Extract<Route, { name: "generate" }> }) {
  const { data: schema } = useSchema(route.tab)
  const desktop = useIsDesktop()
  const layout = useDefaultLayout({ id: "remi-generate-split", panelIds: ["form", "output"] })

  if (!desktop) {
    return (
      <div className="h-full overflow-y-auto">
        <FormColumn schema={schema} />
        <OutputPanel tab={schema.tab} className="p-3 pt-0" />
      </div>
    )
  }

  return (
    <ResizablePanelGroup orientation="horizontal" defaultLayout={layout.defaultLayout} onLayoutChanged={layout.onLayoutChanged}>
      <ResizablePanel id="form" defaultSize="40" minSize={320} maxSize="70">
        <FormColumn schema={schema} />
      </ResizablePanel>
      <ResizableHandle withHandle />
      <ResizablePanel id="output" minSize={320}>
        <OutputPanel tab={schema.tab} className="h-full overflow-y-auto p-3" />
      </ResizablePanel>
    </ResizablePanelGroup>
  )
}
