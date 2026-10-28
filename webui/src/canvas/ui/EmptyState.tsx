/** No image yet: one big drop target (click, drop, paste) and an optional blank canvas. */
import { ImageOff, ImagePlus, Loader2, RotateCw, Square } from "@/components/icons"
import { useId, useRef } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { Size } from "../core/geometry"

type EmptyStateProps = {
  busy: boolean
  noUpload?: boolean
  blankSize?: Size | null
  hint?: string
  error?: string | null
  /** The image the value names could not be loaded. */
  loadFailed?: boolean
  onRetry?: () => void
  onFiles: (files: File[]) => void
  onBlank: () => void
}

/** The value's image did not load: say so and offer to try again (or to pick another below). */
function LoadFailed({ onRetry }: { onRetry?: () => void }) {
  const { t } = useTranslation()
  return (
    <div role="alert" className="flex flex-col items-center gap-2 text-center">
      <span className="grid size-10 place-items-center rounded-full bg-destructive/10 text-destructive">
        <ImageOff className="size-5" aria-hidden />
      </span>
      <p className="text-sm font-medium">{t("Could not load the image")}</p>
      {onRetry ? (
        <Button variant="outline" size="sm" onClick={onRetry}>
          <RotateCw />
          {t("Retry")}
        </Button>
      ) : null}
    </div>
  )
}

export function EmptyState({ busy, noUpload, blankSize, hint, error, loadFailed, onRetry, onFiles, onBlank }: EmptyStateProps) {
  const { t } = useTranslation()
  const input = useRef<HTMLInputElement>(null)
  const hintId = useId()
  if (loadFailed && noUpload) {
    return (
      <div className="flex size-full items-center justify-center p-6">
        <LoadFailed onRetry={onRetry} />
      </div>
    )
  }
  if (noUpload) {
    return (
      <div className="flex size-full flex-col items-center justify-center gap-2 p-6 text-center text-sm text-muted-foreground">
        <ImagePlus className="size-7 opacity-50" aria-hidden />
        <p>{hint ?? t("No image")}</p>
      </div>
    )
  }
  return (
    <div className="flex size-full flex-col items-center justify-center gap-4 overflow-y-auto p-6">
      {loadFailed ? <LoadFailed onRetry={onRetry} /> : null}
      <button
        type="button"
        onClick={() => input.current?.click()}
        aria-describedby={hintId}
        aria-invalid={error ? true : undefined}
        disabled={busy}
        className={cn(
          "group/empty flex w-full max-w-sm flex-col items-center gap-3 rounded-md border border-dashed border-input px-6 py-8 text-center outline-none transition-colors",
          "hover:border-ring/60 hover:bg-primary/5 focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50",
          "aria-invalid:border-destructive/60",
        )}
      >
        <span className="grid size-12 place-items-center rounded-full bg-primary/10 text-primary transition-transform group-hover/empty:scale-105">
          {busy ? <Loader2 className="size-5 animate-spin" aria-hidden /> : <ImagePlus className="size-5" aria-hidden />}
        </span>
        <span className="text-sm font-medium">{busy ? t("Uploading…") : t("Add a source image")}</span>
        <span id={hintId} className="text-xs text-muted-foreground">
          {hint ?? t("Drop an image, paste, or click to browse")}
        </span>
      </button>
      {blankSize ? (
        <Button variant="ghost" size="sm" onClick={onBlank} disabled={busy} className="text-muted-foreground">
          <Square />
          {t("Blank canvas {{width}}×{{height}}", { width: blankSize.width, height: blankSize.height })}
        </Button>
      ) : null}
      {error ? (
        <p role="alert" className="text-sm text-destructive">
          {t(error)}
        </p>
      ) : null}
      <input
        ref={input}
        type="file"
        accept="image/*"
        hidden
        onChange={(event) => {
          const files = Array.from(event.target.files ?? [])
          event.target.value = ""
          if (files.length > 0) onFiles(files)
        }}
      />
    </div>
  )
}
