/**
 * A labelled slider with a number box, for the unit's numeric settings (the
 * renderer's slider look). Like `range-slider`, a dragged value stays local
 * and reaches the unit once, when the drag ends.
 */
import { useId, useState } from "react"
import { NumberField } from "@/components/ui/number-field"
import { Slider } from "@/components/ui/slider"
import { Field } from "@/ui/Field"
import { clamp, snapToStep, stepDecimals } from "@/ui/values"

type SliderRowProps = {
  label: string
  info?: string
  value: number
  minimum: number
  maximum: number
  step: number
  onCommit: (value: number) => void
  disabled?: boolean
}

export function SliderRow({ label, info, value, minimum, maximum, step, onCommit, disabled }: SliderRowProps) {
  const id = useId()
  const decimals = stepDecimals(step)
  // The drag's value, shown while the unit still holds the value it started from
  const [dragged, setDragged] = useState<{ from: number; to: number } | null>(null)
  const shown = dragged !== null && dragged.from === value ? dragged.to : value
  const thumb = (next: number | readonly number[]) => snapToStep(Array.isArray(next) ? next[0]! : (next as number), step, minimum)
  const commit = (raw: number) => {
    if (!Number.isFinite(raw)) return
    const next = clamp(raw, minimum, maximum)
    if (next !== value) onCommit(next)
  }
  return (
    <Field
      id={id}
      label={label}
      info={info}
      disabled={disabled}
      trailing={
        <NumberField
          id={id}
          size="sm"
          steppers={false}
          className="w-[4.75rem]"
          inputClassName="text-right"
          value={shown}
          min={minimum}
          max={maximum}
          step={step}
          format={{ maximumFractionDigits: Math.max(decimals, 0), useGrouping: false }}
          onValueChange={(next) => next !== null && commit(next)}
          disabled={disabled}
        />
      }
    >
      <Slider
        getAriaLabel={() => label}
        value={clamp(shown, minimum, maximum)}
        min={minimum}
        max={maximum}
        step={step}
        onValueChange={(next) => setDragged({ from: value, to: clamp(thumb(next), minimum, maximum) })}
        onValueCommitted={(next) => {
          setDragged(null)
          commit(thumb(next))
        }}
        disabled={disabled}
        // as tall as a select or an input (h-8): a slider beside them lines up its label and its track
        className="flex h-8 items-center"
      />
    </Field>
  )
}
