import { useId } from "react"
import { useTranslation } from "react-i18next"
import { NumberField } from "@/components/ui/number-field"
import { Slider as SliderControl } from "@/components/ui/slider"
import { useParam } from "../hooks"
import { describedBy } from "../a11y"
import { Field } from "../Field"
import type { SliderNode } from "../schema"
import { coerceNumber, snapToStep, stepDecimals } from "../values"

/** Slider with a numeric box in the label row; both edit the same value. */
export function Slider({ node }: { node: SliderNode }) {
  const { t } = useTranslation()
  const id = useId()
  const param = useParam<number>(node)
  const min = param.override.minimum ?? node.minimum
  const max = param.override.maximum ?? node.maximum
  const step = param.override.step ?? node.step
  const integer = node.integer ?? false
  const decimals = integer ? 0 : stepDecimals(step)
  const value = typeof param.value === "number" ? param.value : Number(param.defaultValue ?? min)
  const label = t(param.label)
  const ariaDescribedBy = describedBy(id, param.info, param.error)

  const commit = (raw: number) => {
    if (!Number.isFinite(raw)) return
    const next = coerceNumber(node, raw, { minimum: min, maximum: max })
    if (next !== value) param.setValue(next)
  }

  return (
    <Field
      id={id}
      label={label}
      info={param.info}
      error={param.error}
      modified={param.modified}
      defaultValue={param.defaultValue}
      onReset={param.reset}
      disabled={!param.enabled}
      trailing={
        <NumberField
          id={id}
          aria-label={label ? undefined : node.name}
          aria-describedby={ariaDescribedBy}
          aria-invalid={Boolean(param.error)}
          size="sm"
          steppers={false}
          className="w-[4.75rem]"
          inputClassName="text-right"
          value={value}
          min={min}
          max={max}
          step={step}
          format={{ maximumFractionDigits: Math.max(decimals, 0), useGrouping: false }}
          onValueChange={(next) => next !== null && commit(next)}
          disabled={!param.enabled}
        />
      }
    >
      <SliderControl
        getAriaLabel={() => label || node.name}
        value={value}
        min={min}
        max={max}
        step={step}
        onValueChange={(next) => commit(snapToStep(Array.isArray(next) ? next[0] : next, step, min))}
        disabled={!param.enabled}
        className="py-1.5"
      />
    </Field>
  )
}
