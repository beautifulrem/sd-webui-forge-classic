/** The popover of a chip: its text, weight, enable / disable, favorite, delete. */
import { useQueryClient } from "@tanstack/react-query"
import { Eye, EyeOff, Minus, Plus, RotateCcw, Star, Trash2 } from "lucide-react"
import { useEffect, useId, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { NumberField } from "@/components/ui/number-field"
import { Slider } from "@/components/ui/slider"
import { type PromptToolsConfig, promptToolsApi, type TagInfo } from "../api"
import { categoryName, compactCount } from "../completion/colors"
import { canToggle, formatChip, type PromptDoc, removeChip, setChipText, setChipWeight, toggleChip } from "./parse"

type ChipEditorProps = {
  doc: PromptDoc
  index: number
  info: TagInfo | undefined
  translation: string | undefined
  config: PromptToolsConfig
  setValue: (value: string) => void
  close: () => void
}

const STEP = 0.05
const round = (value: number) => Math.round(value * 100) / 100

export function ChipEditor({ doc, index, info, translation, config, setValue, close }: ChipEditorProps) {
  const { t } = useTranslation()
  const client = useQueryClient()
  const textId = useId()
  const chip = doc.chips[index]
  const [text, setText] = useState(chip?.text ?? "")
  // A weight being dragged or typed: shown here, written into the prompt once
  // (on release, blur or Enter), so one drag is one undo step and one parse.
  const [draft, setDraft] = useState<number | null>(null)
  const pending = useRef<(() => void) | null>(null)
  useEffect(
    () => () => {
      // closed with a typed weight not yet committed
      pending.current?.()
    },
    [],
  )
  if (!chip) return null

  const weightable = chip.kind === "tag" || chip.kind === "network"
  const editable = chip.kind !== "keyword" && chip.kind !== "comment"
  const weight = round(chip.weight)
  const shown = draft ?? weight
  const applyWeight = (next: number) => {
    pending.current = null
    setDraft(null)
    const clamped = round(Math.max(-5, Math.min(5, next)))
    if (clamped !== weight) setValue(setChipWeight(doc, index, clamped))
  }
  const preview = (next: number) => {
    setDraft(next)
    pending.current = () => setValue(setChipWeight(doc, index, round(Math.max(-5, Math.min(5, next)))))
  }
  const applyText = () => {
    if (text.trim() !== chip.text) setValue(setChipText(doc, index, text))
  }
  const category = categoryName(config.tags.source, info?.category)

  const favorite = async () => {
    try {
      await promptToolsApi.favorites.add({ text: formatChip(chip, {}), kind: "tags" })
      await client.invalidateQueries({ queryKey: ["prompt-tools", "favorites"] })
      await client.invalidateQueries({ queryKey: ["prompt-tools", "collections"] })
      toast.success(t("Saved to favorites"))
    } catch (error) {
      toast.error(t("Something went wrong"), { description: error instanceof Error ? error.message : String(error) })
    }
  }

  return (
    <div className="flex flex-col gap-3">
      {editable ? (
        <form
          className="flex flex-col gap-1"
          onSubmit={(event) => {
            event.preventDefault()
            applyText()
            close()
          }}
        >
          <label htmlFor={textId} className="text-xs font-medium text-muted-foreground">
            {chip.kind === "network" ? t("Name") : t("Tag")}
          </label>
          <Input id={textId} value={text} onChange={(event) => setText(event.target.value)} onBlur={applyText} className="h-8" spellCheck={false} />
        </form>
      ) : (
        <p className="text-xs break-words whitespace-pre-wrap text-muted-foreground">{chip.text}</p>
      )}

      {category || info?.count || translation ? (
        <p className="flex flex-wrap gap-x-2 text-xs text-muted-foreground">
          {category ? <span>{category}</span> : null}
          {info?.count ? <span>{t("{{count}} posts", { count: compactCount(info.count) })}</span> : null}
          {translation ? <span className="text-foreground">{translation}</span> : null}
        </p>
      ) : null}

      {weightable ? (
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-1">
            <span className="mr-auto text-xs font-medium text-muted-foreground">{t("Weight")}</span>
            <Button variant="outline" size="icon-xs" aria-label={t("Decrease the weight")} onClick={() => applyWeight(shown - 0.1)}>
              <Minus />
            </Button>
            <NumberField
              value={shown}
              step={STEP}
              min={-5}
              max={5}
              size="sm"
              steppers={false}
              className="w-16"
              inputClassName="text-center"
              aria-label={t("Weight")}
              onValueChange={(next) => {
                if (typeof next === "number" && Number.isFinite(next)) preview(next)
              }}
              onValueCommitted={(next) => {
                if (typeof next === "number" && Number.isFinite(next)) applyWeight(next)
              }}
            />
            <Button variant="outline" size="icon-xs" aria-label={t("Increase the weight")} onClick={() => applyWeight(shown + 0.1)}>
              <Plus />
            </Button>
            <Button variant="ghost" size="icon-xs" aria-label={t("Reset the weight")} disabled={shown === 1} onClick={() => applyWeight(1)}>
              <RotateCcw />
            </Button>
          </div>
          <Slider
            value={Math.max(0, Math.min(2, shown))}
            min={0}
            max={2}
            step={STEP}
            getAriaLabel={() => t("Weight")}
            onValueChange={(next) => {
              const value = Array.isArray(next) ? next[0] : next
              if (typeof value === "number") preview(value)
            }}
            onValueCommitted={(next) => {
              const value = Array.isArray(next) ? next[0] : next
              if (typeof value === "number") applyWeight(value)
            }}
          />
        </div>
      ) : null}

      <div className="flex items-center gap-1 border-t pt-2">
        {config.tools.comments && canToggle(chip) ? (
          <Button variant="ghost" size="sm" onClick={() => setValue(toggleChip(doc, index))}>
            {chip.disabled ? <Eye /> : <EyeOff />}
            {chip.disabled ? t("Enable") : t("Disable")}
          </Button>
        ) : null}
        {editable ? (
          <Button variant="ghost" size="sm" onClick={() => void favorite()}>
            <Star />
            {t("Favorite")}
          </Button>
        ) : null}
        <Button
          variant="ghost"
          size="sm"
          className="ml-auto text-destructive hover:bg-destructive/10 hover:text-destructive"
          onClick={() => {
            close()
            setValue(removeChip(doc, index))
          }}
        >
          <Trash2 />
          {t("Delete")}
        </Button>
      </div>
    </div>
  )
}
