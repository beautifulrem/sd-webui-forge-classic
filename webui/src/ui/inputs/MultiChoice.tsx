import { XIcon } from "@/components/icons"
import { useId, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { COMBOBOX_CHIP_CLASS, COMBOBOX_CHIP_REMOVE_CLASS, COMBOBOX_CHIPS_CLASS, COMBOBOX_CHIPS_INPUT_CLASS } from "@/components/ui/combobox-styles"
import { cn } from "@/lib/utils"
import { lazyModule, prefetch, standInFocus } from "@/lib/lazy-module"
import { useParam } from "../hooks"
import { describedBy, hintRef } from "../a11y"
import { Field } from "../Field"
import type { MultiChoiceNode } from "../schema"
import { useFallbackName } from "../fallback-name"

import type { ChipsSelectProps, Option, Value } from "./chips-select-shared"

// the chips select's combobox code loads when a field is about to be used (see ChipsField)
const chipsSelect = lazyModule(() => import("./ChipsSelect"))

export function MultiChoice({ node }: { node: MultiChoiceNode }) {
  const { t } = useTranslation()
  const id = useId()
  const param = useParam<Value[]>(node)
  const selected = Array.isArray(param.value) ? param.value : []
  const options: Option[] = (param.override.choices ?? node.choices ?? []).map((c) => ({ value: c.value, label: t(c.label) }))
  const label = t(param.label)
  const fallbackName = useFallbackName(node.name)
  const disabled = !param.enabled
  const ariaDescribedBy = describedBy(id, param.info, param.error, hintRef(id, param.hint))

  const control =
    node.style === "checkboxes" ? (
      <div
        role="group"
        aria-labelledby={label ? `${id}-label` : undefined}
        aria-label={label ? undefined : fallbackName}
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
      <ChipsField
        id={id}
        options={options}
        selected={selected}
        onChange={param.setValue}
        allowCustom={node.allow_custom ?? false}
        disabled={disabled}
        ariaLabel={label ? undefined : fallbackName}
        ariaDescribedBy={ariaDescribedBy}
        invalid={Boolean(param.error)}
      />
    )

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
      labelFor={node.style !== "checkboxes"}
    >
      {control}
    </Field>
  )
}

/**
 * The chips select of a dropdown-style multiple choice. Until it is used, a
 * stand-in renders the same chips and search field without the combobox
 * code (`ChipsSelect.tsx`); the pointer entering the field or focus fetches
 * the code, and a press, typing or an opening key puts the real field in the
 * stand-in's place (opened), keyboard focus with the focus: the typed text
 * and the caret carry over. A chip's remove button works in the stand-in too.
 */
function ChipsField(props: Omit<ChipsSelectProps, "query" | "setQuery">) {
  const { t } = useTranslation()
  const module = chipsSelect.useModule()
  const key = useId()
  const [query, setQuery] = useState("")
  const [activation, setActivation] = useState<"open" | "focus" | null>(null)
  // a press focuses the field before its click: that focus must not replace the field under the pointer
  const pressing = useRef(false)
  if (module && activation) return <module.ChipsSelect {...props} query={query} setQuery={setQuery} defaultOpen={activation === "open"} standInKey={key} />

  const activate = (how: "open" | "focus") => {
    setActivation((current) => (current === "open" ? current : how))
    prefetch(chipsSelect)
  }
  const byValue = new Map(props.options.map((o) => [o.value, o]))
  const chips: Option[] = props.selected.map((v) => byValue.get(v) ?? { value: v, label: String(v) })
  return (
    <div role={chips.length ? "toolbar" : undefined} data-slot="combobox-chips" className={cn(COMBOBOX_CHIPS_CLASS, "w-full")} onPointerEnter={() => prefetch(chipsSelect)}>
      {chips.map((option) => (
        <div key={String(option.value)} tabIndex={-1} data-slot="combobox-chip" className={COMBOBOX_CHIP_CLASS}>
          {option.label}
          <Button
            variant="ghost"
            size="icon-xs"
            tabIndex={-1}
            data-slot="combobox-chip-remove"
            aria-disabled={props.disabled}
            disabled={props.disabled}
            className={COMBOBOX_CHIP_REMOVE_CLASS}
            onClick={() => props.onChange(props.selected.filter((v) => v !== option.value))}
          >
            <XIcon className="pointer-events-none" />
          </Button>
        </div>
      ))}
      <input
        autoComplete="off"
        spellCheck={false}
        autoCorrect="off"
        autoCapitalize="none"
        role="combobox"
        aria-expanded={false}
        aria-haspopup="listbox"
        aria-autocomplete="list"
        id={props.id}
        data-slot="combobox-chip-input"
        aria-label={props.ariaLabel}
        aria-describedby={props.ariaDescribedBy}
        aria-invalid={props.invalid || undefined}
        placeholder={chips.length === 0 ? t("Select…") : undefined}
        disabled={props.disabled}
        data-disabled={props.disabled ? "" : undefined}
        className={COMBOBOX_CHIPS_INPUT_CLASS}
        value={query}
        onPointerDown={() => {
          pressing.current = true
        }}
        onFocus={() => {
          standInFocus.focused(key)
          if (pressing.current) prefetch(chipsSelect)
          else activate("focus")
        }}
        onBlur={() => {
          pressing.current = false
          standInFocus.blurred(key)
        }}
        onClick={() => {
          pressing.current = false
          activate("open")
        }}
        onChange={(event) => {
          setQuery(event.target.value)
          activate("open")
        }}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault()
            activate("open")
          }
        }}
      />
    </div>
  )
}
