/** Generation parameters of a result: prompt, negative prompt and parameter chips, copyable. */
import { Check, Copy } from "@/components/icons"
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { copyText } from "@/features/gallery/files"
import { infotextKeyLabel, splitInfotext } from "./display"

function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <Button
      variant="ghost"
      size="icon-xs"
      aria-label={label}
      title={label}
      onClick={() =>
        void copyText(text).then((ok) => {
          if (!ok) return
          setCopied(true)
          setTimeout(() => setCopied(false), 1200)
        })
      }
    >
      {copied ? <Check className="text-success" /> : <Copy />}
    </Button>
  )
}

export function InfotextPanel({ infotext, className }: { infotext: string | null | undefined; className?: string }) {
  const { t } = useTranslation()
  if (!infotext) {
    return <p className={cn("text-xs text-muted-foreground", className)}>{t("No generation parameters")}</p>
  }
  const parts = splitInfotext(infotext)
  return (
    <div className={cn("flex flex-col gap-2.5 text-xs", className)}>
      <div className="flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <h3 className="mb-0.5 text-[0.8rem] font-medium text-muted-foreground">{t("Prompt")}</h3>
          <p className="leading-relaxed break-words whitespace-pre-wrap select-text">{parts.prompt || "—"}</p>
        </div>
        <CopyButton text={infotext} label={t("Copy generation parameters")} />
      </div>
      {parts.negative ? (
        <div>
          <h3 className="mb-0.5 text-[0.8rem] font-medium text-muted-foreground">{t("Negative prompt")}</h3>
          <p className="leading-relaxed break-words whitespace-pre-wrap text-muted-foreground select-text">{parts.negative}</p>
        </div>
      ) : null}
      {parts.params.length > 0 ? (
        <dl className="flex flex-wrap gap-1">
          {parts.params.map(([key, value], index) => (
            <div key={`${key}-${index}`} className="flex max-w-full min-w-0 items-baseline gap-1 rounded-md bg-muted px-1.5 py-0.5">
              <dt className="shrink-0 text-muted-foreground" title={infotextKeyLabel(t, key) === key ? undefined : key}>
                {infotextKeyLabel(t, key)}
              </dt>
              <dd className="min-w-0 truncate font-medium tabular-nums select-text" title={value}>
                {value}
              </dd>
            </div>
          ))}
        </dl>
      ) : null}
    </div>
  )
}
