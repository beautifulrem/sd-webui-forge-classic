/**
 * `block-weights`: one weight per model block, edited as a compact grid of
 * vertical sliders (LoRA block weights, stage multipliers, …).
 *
 * props: { blocks: string[], minimum?: number, maximum?: number, step?: number,
 *          presets?: { label: string, values: number[] }[] }
 * value: number[] (one per block)
 */
import { useTranslation } from "react-i18next"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import type { WidgetProps } from "@/ui/widgets"

type Props = {
  blocks?: string[]
  minimum?: number
  maximum?: number
  step?: number
  presets?: { label: string; values: number[] }[]
}

export default function BlockWeights({ value, onChange, props, disabled, id, label }: WidgetProps<number[], Props>) {
  const { t } = useTranslation()
  const blocks = props.blocks ?? []
  const min = props.minimum ?? 0
  const max = props.maximum ?? 1
  const step = props.step ?? 0.05
  const weights = blocks.map((_, i) => (Array.isArray(value) && typeof value[i] === "number" ? value[i] : 1))

  const set = (index: number, next: number) => onChange(weights.map((w, i) => (i === index ? next : w)))

  return (
    <div id={id} role="group" aria-label={label || t("Block weights")} className="flex flex-col gap-2">
      <div className="flex gap-1 overflow-x-auto rounded-lg border bg-muted/30 p-2">
        {blocks.map((block, i) => (
          <div key={block} className="flex w-9 shrink-0 flex-col items-center gap-1">
            <span className="text-[0.65rem] text-muted-foreground tabular-nums">{(weights[i] ?? 0).toFixed(2)}</span>
            <Slider
              orientation="vertical"
              value={weights[i]}
              min={min}
              max={max}
              step={step}
              disabled={disabled}
              onValueChange={(v) => set(i, Array.isArray(v) ? v[0] : v)}
              getAriaLabel={() => block}
              className="h-24"
            />
            <span className="max-w-full truncate text-[0.65rem] text-muted-foreground" title={block}>
              {block}
            </span>
          </div>
        ))}
      </div>
      {props.presets?.length ? (
        <div className="flex flex-wrap gap-1">
          {props.presets.map((preset) => (
            <Button
              key={preset.label}
              variant="outline"
              size="xs"
              disabled={disabled}
              onClick={() => onChange(blocks.map((_, i) => preset.values[i] ?? 1))}
            >
              {t(preset.label)}
            </Button>
          ))}
        </div>
      ) : null}
    </div>
  )
}
