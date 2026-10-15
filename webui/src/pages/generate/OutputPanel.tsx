/**
 * Result area of one form: live preview while a job queued from this form
 * runs, then the newest result's images and infotext.
 */
import { Copy, ImageIcon } from "lucide-react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import { lastResult, progressFraction, runningFor, useProgress } from "@/api/progress"
import { useWs } from "@/api/ws"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils"

function formatEta(seconds: number | null | undefined): string | null {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return null
  const s = Math.max(0, Math.round(seconds))
  return s >= 60 ? `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}` : `${s}s`
}

export function OutputPanel({ tab, className }: { tab: string; className?: string }) {
  const { t } = useTranslation()
  const running = useProgress((s) => runningFor(s, tab))
  const last = useProgress((s) => lastResult(s, tab))
  // Only the running job's own frames: the previous job's last frame must not flash.
  const preview = useWs((s) => (running && s.preview && (running.seq === undefined || s.preview.jobSeq === running.seq) ? s.preview : null))
  const fraction = progressFraction(running)
  const images = last?.outputs.filter((o) => (o.kind ?? "image") === "image") ?? []
  const [first] = images
  const eta = formatEta(running?.eta)

  return (
    <section aria-label={t("Results")} className={cn("flex min-h-0 flex-col gap-3", className)}>
      <div className="relative flex min-h-64 flex-1 items-center justify-center overflow-hidden rounded-xl border bg-canvas">
        {preview ? (
          <img src={preview.url} alt={t("Live preview")} className="max-h-full max-w-full object-contain" />
        ) : first ? (
          <a href={api.fileUrl(first.id)} target="_blank" rel="noopener noreferrer" className="flex size-full items-center justify-center">
            <img src={api.fileUrl(first.id)} alt={t("Generated image")} className="max-h-full max-w-full object-contain" />
          </a>
        ) : (
          <div className="flex flex-col items-center gap-2 p-6 text-center text-sm text-muted-foreground">
            <ImageIcon className="size-8 opacity-50" aria-hidden />
            <p>{last?.error ? t("Generation failed") : t("Generated images appear here")}</p>
            {last?.error ? <p className="max-w-sm text-xs text-destructive">{last.error}</p> : null}
          </div>
        )}
        {running ? (
          <div className="absolute inset-x-3 bottom-3 rounded-lg bg-background/80 p-2.5 shadow-sm backdrop-blur" role="status" aria-live="polite">
            <div className="mb-1.5 flex items-center justify-between gap-2 text-xs tabular-nums">
              <span className="truncate">
                {running.text ? t(running.text) : t("Step {{step}} of {{steps}}", { step: running.step, steps: running.steps })}
              </span>
              <span className="shrink-0 text-muted-foreground">
                {Math.round((fraction ?? 0) * 100)}%{eta ? ` · ${t("ETA {{eta}}", { eta })}` : ""}
              </span>
            </div>
            <Progress value={(fraction ?? 0) * 100} aria-label={t("Generation progress")} />
          </div>
        ) : null}
      </div>

      {images.length > 1 ? (
        <ul className="grid grid-cols-[repeat(auto-fill,minmax(4.5rem,1fr))] gap-1.5" aria-label={t("Images")}>
          {images.map((image) => (
            <li key={image.id}>
              <a
                href={api.fileUrl(image.id)}
                target="_blank"
                rel="noopener noreferrer"
                className="block overflow-hidden rounded-md bg-checker outline-none focus-visible:ring-3 focus-visible:ring-ring/60"
              >
                <img src={api.thumbUrl(image.id, 192)} alt="" loading="lazy" className="aspect-square w-full object-cover" />
              </a>
            </li>
          ))}
        </ul>
      ) : null}

      {last?.infotext ? (
        <div className="group/info relative rounded-lg border bg-card p-3 pr-10 text-xs leading-relaxed break-words whitespace-pre-wrap text-muted-foreground">
          {last.infotext}
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label={t("Copy generation parameters")}
            className="absolute top-1.5 right-1.5"
            onClick={() => {
              void navigator.clipboard.writeText(last.infotext ?? "").then(() => toast.success(t("Copied")))
            }}
          >
            <Copy />
          </Button>
        </div>
      ) : null}
    </section>
  )
}
