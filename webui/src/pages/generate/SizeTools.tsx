/**
 * Size helpers next to the width/height sliders: aspect presets that keep
 * the current pixel budget, and typing or pasting "W x H" (pasting such
 * text anywhere in the size card sets both).
 */
import { RectangleHorizontal } from "lucide-react"
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { cn } from "@/lib/utils"
import { useFormState } from "@/ui/hooks"
import { ASPECTS, aspectSize, matchAspect, parseSize } from "./size"
import { useSetSize } from "./size-hooks"

export function SizeTools({ className }: { className?: string }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [text, setText] = useState("")
  const width = useFormState((s) => Number(s.values.core?.width ?? 0))
  const height = useFormState((s) => Number(s.values.core?.height ?? 0))
  const setSize = useSetSize()
  const area = Math.max(64 * 64, width * height)
  const current = matchAspect({ width, height })
  const typed = parseSize(text)

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger render={<Button variant="outline" size="sm" className={cn("gap-1.5 tabular-nums", className)} />}>
        <RectangleHorizontal aria-hidden />
        <span className="sr-only">{t("Aspect ratio")}</span>
        {current ? `${current[0]}:${current[1]}` : t("Custom")}
      </PopoverTrigger>
      <PopoverContent align="start" className="w-72">
        <p className="text-xs text-muted-foreground">{t("Aspect presets keep about {{megapixels}} MP", { megapixels: (area / 1e6).toFixed(2) })}</p>
        <div role="group" aria-label={t("Aspect ratio")} className="grid grid-cols-3 gap-1">
          {ASPECTS.map((aspect) => {
            const size = aspectSize(aspect, area)
            const active = current?.[0] === aspect[0] && current[1] === aspect[1]
            return (
              <Button
                key={aspect.join(":")}
                variant={active ? "secondary" : "ghost"}
                aria-pressed={active}
                className="h-auto flex-col gap-0 py-1.5"
                onClick={() => {
                  setSize(size)
                  setOpen(false)
                }}
              >
                <span className="text-sm font-medium tabular-nums">
                  {aspect[0]}:{aspect[1]}
                </span>
                <span className="text-[0.65rem] text-muted-foreground tabular-nums">
                  {size.width}×{size.height}
                </span>
              </Button>
            )
          })}
        </div>
        <form
          className="flex gap-1.5"
          onSubmit={(event) => {
            event.preventDefault()
            if (!typed) return
            setSize(typed)
            setText("")
            setOpen(false)
          }}
        >
          <Input
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="1216x832"
            aria-label={t("Size as width x height")}
            aria-invalid={text !== "" && !typed ? true : undefined}
            className="h-8 tabular-nums"
          />
          <Button type="submit" size="sm" disabled={!typed}>
            {t("Set")}
          </Button>
        </form>
      </PopoverContent>
    </Popover>
  )
}
