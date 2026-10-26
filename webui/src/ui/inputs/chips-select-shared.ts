/** Props of the chips select (`ChipsSelect.tsx`) and its stand-in (`MultiChoice.tsx`). */
export type Value = string | number
export type Option = { value: Value; label: string }

export type ChipsSelectProps = {
  id: string
  options: Option[]
  selected: Value[]
  onChange: (values: Value[]) => void
  allowCustom: boolean
  disabled: boolean
  ariaLabel?: string
  ariaDescribedBy?: string
  invalid: boolean
  /** The search text: kept by the field, so text typed into the stand-in carries over. */
  query: string
  setQuery: (query: string) => void
}
