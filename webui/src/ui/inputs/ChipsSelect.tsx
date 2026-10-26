/**
 * The chips select of a multiple choice (Base UI Combobox). `MultiChoice`
 * renders a stand-in of it (the same chips and search field) and loads this
 * module when the pointer enters the field, it gets focus or the user types,
 * so a form's first load does not carry the combobox code
 * (docs/research/ui-performance.md, "Bundle").
 */
import { useLayoutEffect } from "react"
import { useTranslation } from "react-i18next"
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
import { standInFocus } from "@/lib/lazy-module"
import type { ChipsSelectProps, Option } from "./chips-select-shared"

export function ChipsSelect(props: ChipsSelectProps & { defaultOpen?: boolean; standInKey?: string }) {
  const { t } = useTranslation()
  const anchor = useComboboxAnchor()
  const { query, setQuery } = props
  useLayoutEffect(() => {
    // the stand-in this replaced had the focus: the search field takes it, the caret after the typed text
    if (!props.standInKey || !standInFocus.take(props.standInKey)) return
    const input = anchor.current?.querySelector("input")
    input?.focus()
    input?.setSelectionRange(input.value.length, input.value.length)
    // oxlint-disable-next-line react/exhaustive-deps -- once, when the implementation replaces the stand-in
  }, [])
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
      // opened when the stand-in was pressed or typed into (the focus stays in the field: nothing to hand back)
      defaultOpen={props.defaultOpen}
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
