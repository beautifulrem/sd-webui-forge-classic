import { useId, useState } from "react"
import { useTranslation } from "react-i18next"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Combobox,
  ComboboxChip,
  ComboboxChips,
  ComboboxChipsInput,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxItem,
  ComboboxList,
  ComboboxValue,
  useComboboxAnchor,
} from "@/components/ui/combobox"
import { useParam } from "../hooks"
import { describedBy } from "../a11y"
import { Field } from "../Field"
import type { MultiChoiceNode } from "../schema"

type Value = string | number
type Option = { value: Value; label: string }

export function MultiChoice({ node }: { node: MultiChoiceNode }) {
  const { t } = useTranslation()
  const id = useId()
  const param = useParam<Value[]>(node)
  const selected = Array.isArray(param.value) ? param.value : []
  const options: Option[] = (param.override.choices ?? node.choices ?? []).map((c) => ({ value: c.value, label: t(c.label) }))
  const label = t(param.label)
  const disabled = !param.enabled
  const ariaDescribedBy = describedBy(id, param.info, param.error)

  const control =
    node.style === "checkboxes" ? (
      <div
        role="group"
        aria-labelledby={label ? `${id}-label` : undefined}
        aria-label={label ? undefined : node.name}
        aria-describedby={ariaDescribedBy}
        className="flex flex-wrap gap-x-4 gap-y-2"
      >
        {options.map((option, index) => {
          const optionId = `${id}-option-${index}` // values may contain spaces (id references split on them)
          const checked = selected.includes(option.value)
          return (
            <div key={String(option.value)} className="flex items-center gap-2">
              <Checkbox
                id={optionId}
                checked={checked}
                aria-invalid={param.error ? true : undefined}
                disabled={disabled}
                onCheckedChange={(on) =>
                  param.setValue(
                    on
                      ? // Keep the declared choice order.
                        options.map((o) => o.value).filter((v) => v === option.value || selected.includes(v))
                      : selected.filter((v) => v !== option.value),
                  )
                }
              />
              <label htmlFor={optionId} className="text-sm select-none">
                {option.label}
              </label>
            </div>
          )
        })}
      </div>
    ) : (
      <ChipsSelect
        id={id}
        options={options}
        selected={selected}
        onChange={param.setValue}
        allowCustom={node.allow_custom ?? false}
        disabled={disabled}
        ariaLabel={label ? undefined : node.name}
        ariaDescribedBy={ariaDescribedBy}
        invalid={Boolean(param.error)}
      />
    )

  return (
    <Field
      id={id}
      label={label}
      info={param.info}
      error={param.error}
      modified={param.modified}
      defaultValue={param.defaultValue}
      onReset={param.reset}
      disabled={disabled}
      labelFor={node.style !== "checkboxes"}
    >
      {control}
    </Field>
  )
}

function ChipsSelect(props: {
  id: string
  options: Option[]
  selected: Value[]
  onChange: (values: Value[]) => void
  allowCustom: boolean
  disabled: boolean
  ariaLabel?: string
  ariaDescribedBy?: string
  invalid: boolean
}) {
  const { t } = useTranslation()
  const anchor = useComboboxAnchor()
  const [query, setQuery] = useState("")
  const byValue = new Map(props.options.map((o) => [o.value, o]))
  const selectedOptions = props.selected.map((v) => byValue.get(v) ?? { value: v, label: String(v) })

  const addCustom = () => {
    const text = query.trim()
    if (!props.allowCustom || !text || props.selected.includes(text)) return false
    if (props.options.some((o) => o.label === text)) return false
    props.onChange([...props.selected, text])
    setQuery("")
    return true
  }

  return (
    <Combobox<Option, true>
      multiple
      items={props.options}
      value={selectedOptions}
      onValueChange={(next) => props.onChange(next.map((o) => o.value))}
      inputValue={query}
      onInputValueChange={setQuery}
      itemToStringLabel={(o) => o.label}
      isItemEqualToValue={(a, b) => a.value === b.value}
      disabled={props.disabled}
    >
      <ComboboxChips ref={anchor} className="w-full">
        <ComboboxValue>
          {(values: Option[]) => (
            <>
              {values.map((option) => (
                <ComboboxChip key={String(option.value)}>{option.label}</ComboboxChip>
              ))}
              <ComboboxChipsInput
                id={props.id}
                aria-label={props.ariaLabel}
                aria-describedby={props.ariaDescribedBy}
                aria-invalid={props.invalid || undefined}
                placeholder={values.length === 0 ? t("Select…") : undefined}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && addCustom()) event.preventDefault()
                }}
              />
            </>
          )}
        </ComboboxValue>
      </ComboboxChips>
      <ComboboxContent anchor={anchor}>
        <ComboboxEmpty>{props.allowCustom ? t("Press Enter to add") : t("No matches")}</ComboboxEmpty>
        <ComboboxList>
          {(option: Option) => (
            <ComboboxItem key={String(option.value)} value={option}>
              {option.label}
            </ComboboxItem>
          )}
        </ComboboxList>
      </ComboboxContent>
    </Combobox>
  )
}
