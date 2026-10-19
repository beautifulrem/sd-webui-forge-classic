/**
 * Extras: upscale and post-process one image, a batch of uploads or a folder
 * on the server (the form and every postprocessing operation are rendered
 * from the server's schema). The result shows with a before / after divider
 * against its source, with download, send-to and the viewer; this session's
 * runs stay in a strip below.
 */
import { Ban, ImageIcon, ImageUpscale, Loader2, Maximize2, Sparkles } from "lucide-react"
import { lazy, Suspense, useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { api } from "@/api/endpoints"
import { progressFraction, runningFor, useProgress } from "@/api/progress"
import { useSchema } from "@/api/queries"
import { useWs } from "@/api/ws"
import { formatShortcut } from "@/app/commands"
import { Segmented } from "@/components/Segmented"
import { Button } from "@/components/ui/button"
import { Kbd } from "@/components/ui/kbd"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"
import { useIsDesktop } from "@/lib/use-media-query"
import { cn } from "@/lib/utils"
import { CompareSlider } from "@/features/compare/CompareSlider"
import { type ExtrasResult, interruptExtras, runExtras, useExtrasRuns, wireExtrasRuns } from "@/features/extras/run"
import type { ResultItem } from "@/features/gallery/session-store"
import { formatEta } from "@/features/progress/effects"
import { enqueueForm } from "@/features/queue/edit-session"
import { EnqueueButton, QueueEditBanner } from "@/features/queue/FormQueueControls"
import { ResultActions } from "@/features/send-to/ResultActions"
import { useFormLoader } from "@/pages/generate/use-form-loader"
import { FormProvider, OwnerProvider } from "@/ui/context"
import { DEFAULT_SLOT } from "@/ui/form-index"
import { getFormStore } from "@/ui/form-store"
import { ScriptsProvider, Slot } from "@/ui/layout/Slot"
import { RenderLayout } from "@/ui/Renderer"

const Lightbox = lazy(() => import("@/features/lightbox/Lightbox").then((m) => ({ default: m.Lightbox })))

wireExtrasRuns()

function ExtrasForm() {
  const { t } = useTranslation()
  const { data: schema } = useSchema("extras")
  const store = getFormStore("extras")
  const ready = useFormLoader(store, schema)
  if (!ready) return <FormSkeleton />
  const hasDefaultSlot = store.getState().index.slots.has(DEFAULT_SLOT)
  return (
    <FormProvider formId="extras" store={store}>
      <ScriptsProvider scripts={schema.scripts}>
        <div role="form" aria-label={t("Parameters")} className="flex flex-col gap-3">
          <OwnerProvider owner="core">
            <RenderLayout layout={schema.core.layout} />
            {hasDefaultSlot ? null : <Slot section={DEFAULT_SLOT} />}
          </OwnerProvider>
        </div>
      </ScriptsProvider>
    </FormProvider>
  )
}

function FormSkeleton() {
  return (
    <div className="flex flex-col gap-3" aria-busy>
      <Skeleton className="h-56 rounded-xl" />
      <Skeleton className="h-32 rounded-xl" />
      <Skeleton className="h-24 rounded-xl" />
    </div>
  )
}

function RunBar() {
  const { t } = useTranslation()
  const submitting = useExtrasRuns((s) => s.submitting)
  const waiting = useExtrasRuns((s) => s.jobs.length)
  const job = useProgress((s) => runningFor(s, "extras"))
  const fraction = progressFraction(job)
  const eta = formatEta(job?.eta)
  const busy = waiting > 0
  return (
    <div className="flex flex-col gap-2">
      <QueueEditBanner tab="extras" />
      <div className="flex items-center gap-1.5">
        <Button size="lg" className="h-10 min-w-0 flex-1 gap-2 text-sm font-semibold shadow-sm" onClick={() => void runExtras()} disabled={submitting} aria-keyshortcuts="Control+Enter Meta+Enter">
          {submitting ? <Loader2 className="animate-spin" aria-hidden /> : <Sparkles aria-hidden />}
          <span className="truncate">{busy ? t("Queue another") : t("Process")}</span>
          <Kbd className="ml-1 hidden border border-primary-foreground/45 bg-transparent text-primary-foreground sm:inline-flex">{formatShortcut("Mod+Enter")}</Kbd>
        </Button>
        <EnqueueButton tab="extras" />
        {busy ? (
          <Button variant="destructive" size="icon-lg" className="size-10" onClick={interruptExtras} aria-label={t("Interrupt")} title={`${t("Interrupt")} (Esc)`} aria-keyshortcuts="Escape">
            <Ban />
          </Button>
        ) : null}
      </div>
      {busy ? (
        <div className="flex flex-col gap-1.5" role="status">
          {job && fraction !== null ? <Progress value={fraction * 100} aria-label={t("Processing progress")} className="h-1" /> : null}
          <p className="text-xs text-muted-foreground tabular-nums">
            {job
              ? [job.text ? t(job.text) : t("Processing"), job.jobCount && job.jobCount > 1 ? t("Image {{index}} of {{count}}", { index: (job.jobNo ?? 0) + 1, count: job.jobCount }) : null, eta ? t("ETA {{eta}}", { eta }) : null]
                  .filter(Boolean)
                  .join(" · ")
              : t("Queued")}
          </p>
        </div>
      ) : null}
    </div>
  )
}

function asItems(result: ExtrasResult): ResultItem[] {
  return result.outputs
    .filter((output) => (output.kind ?? "image") === "image")
    .map((output, index) => ({
      key: `${result.jobId}:${index}`,
      fileId: output.id,
      jobId: result.jobId,
      tab: "extras",
      index,
      infotext: output.infotext ?? result.info,
      width: output.width ?? undefined,
      height: output.height ?? undefined,
      createdAt: result.at,
    }))
}

function ResultStage() {
  const { t } = useTranslation()
  const results = useExtrasRuns((s) => s.results)
  const selected = useExtrasRuns((s) => s.selected)
  const result = (selected ? results.find((item) => item.jobId === selected) : undefined) ?? results[0] ?? null
  const items = result ? asItems(result) : []
  const [current, setCurrent] = useState<string | null>(null)
  const item = items.find((entry) => entry.key === current) ?? items[0] ?? null
  const [mode, setMode] = useState<"compare" | "result">("compare")
  const [viewer, setViewer] = useState(false)
  const job = useProgress((s) => runningFor(s, "extras"))
  const preview = useWs((s) => (job && s.preview && (job.seq === undefined || s.preview.jobSeq === job.seq) ? s.preview.url : null))
  const comparable = Boolean(result?.source && item && items.length === 1)

  let view: React.ReactNode
  if (job && preview && !item) {
    view = <img src={preview} alt={t("Live preview")} className="max-h-full max-w-full rounded-lg object-contain" />
  } else if (result?.error) {
    view = <p className="max-w-md rounded-lg bg-destructive/10 px-4 py-3 text-sm text-destructive">{result.error}</p>
  } else if (!item) {
    view = (
      <div className="flex flex-col items-center gap-2 text-center text-sm text-muted-foreground">
        <ImageUpscale className="size-8" aria-hidden />
        <p>{result ? t("This run made no image (a folder run saves to its output folder).") : t("Results appear here, next to their source.")}</p>
      </div>
    )
  } else if (comparable && mode === "compare") {
    view = (
      <CompareSlider
        key={item.key}
        before={api.fileUrl(result!.source!)}
        after={api.fileUrl(item.fileId)}
        beforeLabel={t("Source")}
        afterLabel={t("Result")}
        aspect={item.width && item.height ? item.width / item.height : undefined}
        fitHeight="calc(100dvh - 15rem)"
      />
    )
  } else {
    view = (
      <button type="button" className="contents" onClick={() => setViewer(true)} aria-label={t("Open in the viewer")}>
        <img src={api.fileUrl(item.fileId)} alt={t("Processed image")} className="max-h-full max-w-full cursor-zoom-in rounded-lg bg-checker object-contain" />
      </button>
    )
  }

  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      <div className="flex min-h-9 flex-wrap items-center gap-2">
        {comparable ? (
          <Segmented
            size="sm"
            aria-label={t("View")}
            value={mode}
            onValueChange={setMode}
            options={[
              { value: "compare", label: t("Compare") },
              { value: "result", label: t("Result") },
            ]}
          />
        ) : null}
        {item?.width && item.height ? <span className="text-xs text-muted-foreground tabular-nums">{item.width}×{item.height}</span> : null}
        <div className="ml-auto flex items-center gap-0.5">
          {item ? (
            <Button variant="ghost" size="icon-sm" aria-label={t("Open in the viewer")} onClick={() => setViewer(true)}>
              <Maximize2 />
            </Button>
          ) : null}
          <ResultActions item={item} tab="txt2img" removable={false} />
        </div>
      </div>
      <div className="relative flex min-h-0 flex-1 items-center justify-center rounded-xl border bg-canvas/40 p-2">{view}</div>
      {items.length > 1 ? (
        <div role="listbox" aria-label={t("Images of this run")} className="flex gap-1.5 overflow-x-auto pb-1">
          {items.map((entry) => (
            <button
              key={entry.key}
              type="button"
              role="option"
              aria-selected={entry.key === item?.key}
              onClick={() => setCurrent(entry.key)}
              className={cn("size-16 shrink-0 overflow-hidden rounded-md bg-checker outline-none focus-visible:ring-3 focus-visible:ring-ring", entry.key === item?.key && "ring-2 ring-primary")}
            >
              <img src={api.thumbUrl(entry.fileId, 128)} alt="" className="size-full object-cover" />
            </button>
          ))}
        </div>
      ) : null}
      {result?.info ? <p className="line-clamp-3 text-xs break-words text-muted-foreground select-text">{result.info}</p> : null}
      <RecentRuns />
      {viewer && item ? (
        <Suspense fallback={null}>
          <Lightbox open onOpenChange={setViewer} items={items} current={item.key} onNavigate={setCurrent} tab="extras" />
        </Suspense>
      ) : null}
    </div>
  )
}

function RecentRuns() {
  const { t } = useTranslation()
  const results = useExtrasRuns((s) => s.results)
  const selected = useExtrasRuns((s) => s.selected) ?? results[0]?.jobId
  if (results.length < 2) return null
  return (
    <div className="flex flex-col gap-1">
      <h2 className="text-[0.7rem] font-semibold tracking-wide text-muted-foreground uppercase">{t("This session")}</h2>
      <div role="listbox" aria-label={t("Recent runs")} className="flex gap-1.5 overflow-x-auto pb-1">
        {results.map((result) => {
          const first = result.outputs.find((output) => (output.kind ?? "image") === "image")
          return (
            <button
              key={result.jobId}
              type="button"
              role="option"
              aria-selected={result.jobId === selected}
              aria-label={first ? t("Images of this run: {{count}}", { count: result.outputs.length }) : t("Run without images")}
              onClick={() => useExtrasRuns.setState({ selected: result.jobId })}
              className={cn("grid size-12 shrink-0 place-items-center overflow-hidden rounded-md bg-muted outline-none focus-visible:ring-3 focus-visible:ring-ring", result.jobId === selected && "ring-2 ring-primary")}
            >
              {first ? <img src={api.thumbUrl(first.id, 96)} alt="" className="size-full object-cover" /> : result.error ? <Ban className="size-4 text-destructive" aria-hidden /> : <ImageIcon className="size-4 text-muted-foreground" aria-hidden />}
            </button>
          )
        })}
      </div>
    </div>
  )
}

function useExtrasShortcuts() {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.isComposing || event.repeat) return
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        if (document.querySelector("[role=dialog][data-open], [role=alertdialog]")) return
        event.preventDefault()
        // Ctrl/Cmd+Shift+Enter: "Add to queue", as on the Generate page
        if (event.shiftKey) void enqueueForm("extras")
        else void runExtras()
      } else if (event.key === "Escape" && useExtrasRuns.getState().jobs.length > 0) {
        const target = event.target instanceof Element ? event.target : null
        if (target?.closest("input, textarea, [contenteditable], [role=dialog], [role=menu], [role=listbox]")) return
        interruptExtras()
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])
}

export default function ExtrasPage() {
  const { t } = useTranslation()
  const desktop = useIsDesktop()
  useExtrasShortcuts()
  return (
    <div className={cn("h-full", desktop ? "grid grid-cols-[minmax(22rem,30rem)_minmax(0,1fr)]" : "overflow-y-auto")}>
      <div className={cn("flex flex-col", desktop && "min-h-0 border-r")}>
        <div className="sticky top-0 z-10 border-b bg-background/95 p-3 backdrop-blur">
          <RunBar />
        </div>
        <div className={cn("p-3", desktop && "min-h-0 flex-1 overflow-y-auto")}>
          <Suspense fallback={<FormSkeleton />}>
            <ExtrasForm />
          </Suspense>
        </div>
      </div>
      <section aria-label={t("Result")} className={cn("p-3", desktop ? "min-h-0" : "h-[80dvh]")}>
        <ResultStage />
      </section>
    </div>
  )
}
