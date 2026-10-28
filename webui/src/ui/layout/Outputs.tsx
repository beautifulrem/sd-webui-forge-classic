import { CircleAlert, CircleCheck, CircleX, ImageOff, Info } from "@/components/icons"
import { useTranslation } from "react-i18next"
import { cn } from "@/lib/utils"
import { useFormState, useOwner } from "../hooks"
import { SafeHtml } from "../RichText"
import { type ImageRef, toImageRef } from "../image-ref"
import { isMessage, messageText } from "../messages"
import type { GalleryNode, OutputNode } from "../schema"
import { parseStatus, type StatusTone } from "../status-text"

const TONES: Record<StatusTone, { icon: typeof Info; className: string }> = {
  success: { icon: CircleCheck, className: "bg-success/10 text-foreground [&>svg]:text-success" },
  info: { icon: Info, className: "bg-muted/60 text-foreground [&>svg]:text-muted-foreground" },
  warning: { icon: CircleAlert, className: "bg-warning/12 text-foreground [&>svg]:text-warning" },
  error: { icon: CircleX, className: "bg-destructive/10 text-foreground [&>svg]:text-destructive" },
}

/** A text output; a leading status emoji becomes the shared status styling (icon and tone). */
function TextOutput({ text }: { text: string }) {
  const status = parseStatus(text)
  if (!status.tone) return <div className="rounded-md bg-muted/60 px-2.5 py-2 text-sm whitespace-pre-wrap">{text}</div>
  const tone = TONES[status.tone]
  const Icon = tone.icon
  return (
    <div data-tone={status.tone} className={cn("flex items-start gap-2 rounded-md px-2.5 py-2 text-sm", tone.className)}>
      <Icon aria-hidden className="mt-0.5 size-4 shrink-0" />
      <span className="min-w-0 break-words whitespace-pre-wrap">{status.text}</span>
    </div>
  )
}

function useOutput(name: string): unknown {
  const owner = useOwner()
  return useFormState((s) => s.outputs[owner]?.[name])
}

function OutputLabel({ label }: { label?: string | null }) {
  const { t } = useTranslation()
  return label ? <div className="text-[0.8rem] font-medium text-foreground/85">{t(label)}</div> : null
}

export function GalleryView({ node }: { node: GalleryNode }) {
  const { t } = useTranslation()
  const value = useOutput(node.name)
  const images = (Array.isArray(value) ? value : value ? [value] : []).map(toImageRef).filter((x): x is ImageRef => x !== null)

  return (
    <div className="flex flex-col gap-1.5">
      <OutputLabel label={node.label} />
      {images.length === 0 ? (
        <div className="flex aspect-[3/2] max-h-60 flex-col items-center justify-center gap-2 rounded-lg border border-dashed text-sm text-muted-foreground">
          <ImageOff className="size-5" aria-hidden />
          {t("No images yet")}
        </div>
      ) : (
        <ul className="grid grid-cols-[repeat(auto-fill,minmax(7rem,1fr))] gap-1.5" aria-label={node.label ? t(node.label) : t("Images")}>
          {images.map((image, i) => (
            <li key={`${image.src}-${i}`} className="overflow-hidden rounded-md bg-checker">
              <a href={image.src} target="_blank" rel="noopener noreferrer" className="block outline-none focus-visible:ring-3 focus-visible:ring-ring/60">
                <img src={image.src} alt={image.caption ?? ""} loading="lazy" className="aspect-square w-full object-cover" />
              </a>
              {image.caption ? <p className="truncate px-1.5 py-1 text-xs text-muted-foreground">{image.caption}</p> : null}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export function OutputView({ node }: { node: OutputNode }) {
  const { t, i18n } = useTranslation()
  const value = useOutput(node.name)
  let body
  switch (node.kind ?? "text") {
    case "html":
      body = typeof value === "string" && value ? <SafeHtml html={value} /> : null
      break
    case "image": {
      const ref = toImageRef(value)
      body = ref ? <img src={ref.src} alt={ref.caption ?? ""} className="max-h-96 rounded-md bg-checker object-contain" /> : null
      break
    }
    case "json":
      body =
        value === undefined ? null : (
          <pre className="max-h-80 overflow-auto rounded-md bg-muted p-2.5 font-mono text-xs">{JSON.stringify(value, null, 2)}</pre>
        )
      break
    default: {
      const text = value === undefined || value === null ? "" : isMessage(value) || typeof value === "string" ? messageText(i18n, value) : String(value)
      body = text ? <TextOutput text={text} /> : null
    }
  }
  if (!body && !node.label) return null
  return (
    <div className="flex flex-col gap-1.5" aria-live="polite">
      <OutputLabel label={node.label} />
      {body ?? <div className="rounded-md border border-dashed px-2.5 py-2 text-sm text-muted-foreground">{t("No output yet")}</div>}
    </div>
  )
}
