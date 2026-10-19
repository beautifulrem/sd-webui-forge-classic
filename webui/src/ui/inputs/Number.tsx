import { useId } from "react"
import { useTranslation } from "react-i18next"
import { NumberField } from "@/components/ui/number-field"
import { useParam } from "../hooks"
import { describedBy, hintRef } from "../a11y"
import { Field } from "../Field"
import type { NumberNode } from "../schema"
import { coerceNumber, stepDecimals } from "../values"
import { useFallbackName } from "../fallback-name"

export function NumberInput({ node }: { node: NumberNode }) {
  const { t } = useTranslation()
  const id = useId()
  const param = useParam<number | null>(node)
  const step = param.override.step ?? node.step ?? (node.integer ? 1 : undefined)
  const decimals = node.integer ? 0 : (node.precision ?? (step ? stepDecimals(step) : 6))
  const info = param.info
  const label = t(param.label)
  const fallbackName = useFallbackName(node.name)
  const minimum = param.override.minimum ?? node.minimum
  const maximum = param.override.maximum ?? node.maximum

  return (
    <Field
      id={id}
      label={label}
      info={info}
      hint={param.hint}
      error={param.error}
      modified={param.modified}
      defaultValue={param.defaultValue}
      onReset={param.reset}
      disabled={!param.enabled}
    >
      <NumberField
        id={id}
        aria-label={label ? undefined : fallbackName}
        aria-describedby={describedBy(id, info, param.error, hintRef(id, param.hint))}
        aria-invalid={Boolean(param.error)}
        value={typeof param.value === "number" ? param.value : null}
        min={minimum ?? undefined}
        max={maximum ?? undefined}
        step={step}
        format={{ maximumFractionDigits: decimals, useGrouping: false }}
        onValueChange={(next) => {
          if (next === null) return
          param.setValue(coerceNumber(node, next, { minimum, maximum }))
        }}
        disabled={!param.enabled}
      />
    </Field>
  )
}
