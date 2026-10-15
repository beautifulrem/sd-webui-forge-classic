/** Brush colour: swatch button with a popover (picker, hex, swatches, recent, eyedropper). */
import { Pipette } from "lucide-react"
import { useId, useState } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { cn } from "@/lib/utils"
import { parseHex, toHex } from "../core/pixels"
import { SWATCHES } from "../state/tools"
import { Hint } from "./controls"

type EyeDropperApi = { open: () => Promise<{ sRGBHex: string }> }

function Swatch({ color, selected, onPick, label }: { color: string; selected: boolean; onPick: (color: string) => void; label: string }) {
  return (
    <button
      type="button"
      aria-label={label}
      aria-pressed={selected}
      onClick={() => onPick(color)}
      className={cn(
        "size-6 rounded-md border border-black/10 shadow-xs outline-none transition-transform hover:scale-110 focus-visible:ring-3 focus-visible:ring-ring/50 dark:border-white/15",
        selected && "ring-2 ring-primary ring-offset-1 ring-offset-popover",
      )}
      style={{ backgroundColor: color }}
    />
  )
}

export function ColorControl({ color, recent, onChange, onEyedropper }: { color: string; recent: readonly string[]; onChange: (color: string) => void; onEyedropper: () => void }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [text, setText] = useState<string | null>(null)
  const id = useId()
  const screenPicker = typeof window !== "undefined" && "EyeDropper" in window

  const commitText = (value: string) => {
    const rgb = parseHex(value)
    if (rgb) onChange(toHex(rgb))
    setText(null)
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <Hint label={t("Brush color")} disabled={open}>
        <PopoverTrigger
          render={
            <Button variant="ghost" size="icon-sm" aria-label={t("Brush color")}>
              <span className="size-4.5 rounded-full border border-black/15 shadow-xs dark:border-white/20" style={{ backgroundColor: color }} />
            </Button>
          }
        />
      </Hint>
      <PopoverContent className="w-60" side="bottom" align="start">
        <div className="flex items-center gap-2">
          <label htmlFor={`${id}-native`} className="sr-only">
            {t("Pick a color")}
          </label>
          <input
            id={`${id}-native`}
            type="color"
            value={color}
            onChange={(event) => onChange(event.target.value)}
            className="h-9 w-12 shrink-0 cursor-pointer rounded-md border bg-transparent p-0.5"
          />
          <Input
            aria-label={t("Hex color")}
            value={text ?? color}
            spellCheck={false}
            onChange={(event) => setText(event.target.value)}
            onBlur={(event) => commitText(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") commitText(event.currentTarget.value)
            }}
            aria-invalid={text !== null && parseHex(text) === null ? true : undefined}
            className="h-9 font-mono text-xs uppercase"
          />
          <Hint label={t("Pick a color from the image")} keys={["I"]}>
            <Button
              variant="outline"
              size="icon"
              aria-label={t("Pick a color from the image")}
              onClick={() => {
                setOpen(false)
                onEyedropper()
              }}
            >
              <Pipette />
            </Button>
          </Hint>
        </div>
        <div role="group" aria-label={t("Swatches")} className="grid grid-cols-6 gap-1.5">
          {SWATCHES.map((swatch) => (
            <Swatch key={swatch} color={swatch} selected={swatch === color} onPick={onChange} label={swatch} />
          ))}
        </div>
        {recent.length > 0 ? (
          <div className="flex flex-col gap-1.5">
            <span className="text-xs text-muted-foreground">{t("Recent colors")}</span>
            <div role="group" aria-label={t("Recent colors")} className="grid grid-cols-8 gap-1">
              {recent.map((swatch) => (
                <Swatch key={swatch} color={swatch} selected={swatch === color} onPick={onChange} label={swatch} />
              ))}
            </div>
          </div>
        ) : null}
        {screenPicker ? (
          <Button
            variant="ghost"
            size="sm"
            className="justify-start text-muted-foreground"
            onClick={async () => {
              const dropper = new (window as unknown as { EyeDropper: new () => EyeDropperApi }).EyeDropper()
              setOpen(false)
              try {
                const picked = await dropper.open()
                const rgb = parseHex(picked.sRGBHex)
                if (rgb) onChange(toHex(rgb))
              } catch {
                // Cancelled with Escape.
              }
            }}
          >
            <Pipette />
            {t("Pick from anywhere on screen")}
          </Button>
        ) : null}
      </PopoverContent>
    </Popover>
  )
}
