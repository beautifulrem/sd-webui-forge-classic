import { useId, useRef } from "react"
import { useTranslation } from "react-i18next"
import { SearchSelect } from "@/components/SearchSelect"
import { Segmented } from "@/components/Segmented"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useParam } from "../hooks"
import { describedBy, hintRef } from "../a11y"
import { Field } from "../Field"
import type { ChoiceNode, ChoiceOption } from "../schema"
import { useFallbackName } from "../fallback-name"
import { useSegmentsFit } from "./segment-fit"

/** Up to this many short options render as a segmented control (radio style). */
const SEGMENTED_MAX = 5
const SEGMENTED_LABEL_MAX = 14
/** Longer dropdown lists get a search field. */
const PLAIN_SELECT_MAX = 12

type Value = string | number

export function Choice({ node }: { node: ChoiceNode }) {
  const { t } = useTranslation()
  const id = useId()
  const param = useParam<Value | null>(node)
  const choices: ChoiceOption[] = param.override.choices ?? node.choices ?? []
  // choice tooltips are keyed by the untranslated label
  const options = choices.map((c) => ({ value: c.value, label: t(c.label), hint: node.choice_hints?.[c.label] ? t(node.choice_hints[c.label]!) : undefined }))
  const label = t(param.label)
  const fallbackName = useFallbackName(node.name)
  const labelId = `${id}-label`
  const value = param.value ?? null
  const disabled = !param.enabled
  const aria = { "aria-describedby": describedBy(id, param.info, param.error, hintRef(id, param.hint)), "aria-invalid": param.error ? true : undefined }
  const radio = node.style === "radio"
  const fitRef = useRef<HTMLDivElement>(null)
  const short = radio && options.length <= SEGMENTED_MAX && options.every((o) => o.label.length <= SEGMENTED_LABEL_MAX)
  // the translated labels must fit their segments, else radios (no ellipsised option)
  const fits = useSegmentsFit(
    fitRef,
    options.map((o) => o.label),
    short,
  )

  let control
  if (radio) {
    control = short && fits ? (
      <Segmented
        options={options}
        value={value}
        onValueChange={param.setValue}
        disabled={disabled}
        aria-labelledby={label ? labelId : undefined}
        aria-label={label ? undefined : fallbackName}
        {...aria}
        className="w-full"
      />
    ) : (
      <RadioGroup
        value={value}
        onValueChange={(next) => param.setValue(next as Value)}
        disabled={disabled}
        aria-labelledby={label ? labelId : undefined}
        aria-label={label ? undefined : fallbackName}
        {...aria}
        className="flex flex-wrap gap-x-4 gap-y-2"
      >
        {options.map((option, index) => {
          // index-based: values may contain spaces, which break id references (aria-labelledby)
          const optionId = `${id}-option-${index}`
          return (
            <div key={String(option.value)} className="flex items-center gap-2">
              <RadioGroupItem id={optionId} value={option.value} />
              <label htmlFor={optionId} title={option.hint} className="text-sm select-none">
                {option.label}
              </label>
            </div>
          )
        })}
      </RadioGroup>
    )
  } else if (options.length <= PLAIN_SELECT_MAX && !node.allow_custom) {
    control = (
      <Select<Value>
        items={options}
        value={value}
        onValueChange={(next) => next !== null && param.setValue(next)}
        disabled={disabled}
      >
        <SelectTrigger id={id} aria-label={label ? undefined : fallbackName} {...aria} className="w-full">
          <SelectValue placeholder={t("Select…")} />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={String(option.value)} value={option.value} title={option.hint}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    )
  } else {
    // SearchSelect works on strings; map numeric values through their string form.
    const byString = new Map(options.map((o) => [String(o.value), o.value]))
    control = (
      <SearchSelect
        id={id}
        options={options.map((o) => ({ value: String(o.value), label: o.label }))}
        value={value === null ? null : String(value)}
        onValueChange={(next) => param.setValue(byString.get(next) ?? next)}
        allowCustom={node.allow_custom}
        disabled={disabled}
        aria-label={label ? undefined : fallbackName}
        {...aria}
      />
    )
  }

  return (
    <Field
      id={id}
      label={label}
      info={param.info}
      hint={param.hint}
      error={param.error}
      modified={param.modified}
      defaultValue={param.defaultValue}
      onReset={param.reset}
      disabled={disabled}
      labelFor={node.style !== "radio"}
    >
      {radio ? (
        <div ref={fitRef} className="min-w-0">
          {control}
        </div>
      ) : (
        control
      )}
    </Field>
  )
}
