/** Training-dataset tags as chips weighted by frequency (most frequent first). */
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { tagWeight, type WeightedTag } from "./tags"

const INITIAL = 60

export function TagChips({
  tags,
  onPick,
  isOn,
  pickLabel,
  className,
}: {
  tags: readonly WeightedTag[]
  onPick: (tag: string) => void
  isOn?: (tag: string) => boolean
  /** Accessible description of what clicking a chip does. */
  pickLabel: string
  className?: string
}) {
  const { t } = useTranslation()
  const [all, setAll] = useState(false)
  const max = tags[0]?.count ?? 0
  const shown = all ? tags : tags.slice(0, INITIAL)
  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <ul aria-label={pickLabel} className="flex flex-wrap gap-1.5">
        {shown.map(({ tag, count }) => {
          const weight = tagWeight(count, max)
          const on = isOn?.(tag) ?? false
          return (
            <li key={tag}>
              <button
                type="button"
                aria-pressed={isOn ? on : undefined}
                onClick={() => onPick(tag)}
                title={t("Images: {{count}}", { count })}
                className={cn(
                  "inline-flex h-6 max-w-64 items-center gap-1.5 rounded-full border px-2.5 text-xs transition-colors outline-none",
                  "focus-visible:ring-2 focus-visible:ring-ring/60",
                  on ? "border-primary bg-primary text-primary-foreground" : "border-transparent hover:border-ring/40",
                )}
                style={on ? undefined : { backgroundColor: `color-mix(in oklch, var(--primary) ${Math.round(6 + weight * 26)}%, var(--muted))` }}
              >
                <span className="truncate" style={{ fontWeight: weight > 0.6 ? 600 : weight > 0.3 ? 500 : 400 }}>
                  {tag}
                </span>
                <span className={cn("tabular-nums", on ? "text-primary-foreground" : "text-foreground/70")}>{count}</span>
              </button>
            </li>
          )
        })}
      </ul>
      {tags.length > INITIAL ? (
        <Button variant="ghost" size="xs" className="self-start" onClick={() => setAll(!all)}>
          {all ? t("Show fewer") : t("Show all {{count}} tags", { count: tags.length })}
        </Button>
      ) : null}
    </div>
  )
}
