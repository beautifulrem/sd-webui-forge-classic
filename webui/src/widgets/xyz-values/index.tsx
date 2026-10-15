/**
 * `xyz-values`: the X/Y/Z plot's values dropdown. Its options are the values
 * of the axis selected in the sibling param `props.axis` (`x_type`, …).
 *
 * - One call of the script action `props.action` returns every list axis'
 *   values (`{label: values | null}`, null: that axis' list failed), kept in
 *   the form's outputs until the schema reloads (a checkpoint refresh reloads
 *   it). The dropdowns of one form share a single in-flight call.
 * - The axis actions (`axis_x`, `fill_x`, …) return the selected axis' fresh
 *   list as the output `axis_choices/<label>`, which wins over the cached one,
 *   as the old UI re-read the list on every axis change.
 * - After a failure the call is retried when the axis changes or the
 *   dropdown opens.
 *
 * Values outside the current options stay as chips; the script checks them
 * when it runs.
 *
 * props: { axis: string, action?: string }   value: string[]
 */
import { useEffect, useState } from "react"
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
import type { FormStore } from "@/ui/form-store"
import { useForm, useFormState, useOwner, useRunAction } from "@/ui/hooks"
import type { WidgetProps } from "@/ui/widgets"
import { registerWidgetCatalogs } from "../_lib/i18n"

const NS = registerWidgetCatalogs("xyz-values", import.meta.glob("./locales/*.json", { eager: true, import: "default" }))

/** Output key of the action's answer: `{axis label: values | null}`; `${CHOICES_OUTPUT}/${label}` holds one axis' fresh list. */
const CHOICES_OUTPUT = "axis_choices"

type Props = { axis: string; action?: string }
type AxisChoices = Record<string, string[] | null>

const EMPTY: string[] = []

/** In-flight loads per form store and `owner/action`: the X, Y and Z dropdowns share one. */
const inflight = new WeakMap<FormStore, Map<string, Promise<boolean>>>()

function shared(store: FormStore, key: string, start: () => Promise<boolean>): Promise<boolean> {
  let loads = inflight.get(store)
  if (!loads) inflight.set(store, (loads = new Map()))
  let load = loads.get(key)
  if (!load) {
    load = start().finally(() => loads.delete(key))
    loads.set(key, load)
  }
  return load
}

export default function XyzValues({ value, onChange, props, disabled, id, label }: WidgetProps<string[] | null, Props>) {
  const { t: tw } = useTranslation(NS)
  const { store } = useForm()
  const owner = useOwner()
  const run = useRunAction()
  const anchor = useComboboxAnchor()
  const [query, setQuery] = useState("")
  const [pending, setPending] = useState(false)
  const [failed, setFailed] = useState(false)
  const axis = String(useFormState((s) => s.values[owner]?.[props.axis]))
  const all = useFormState((s) => s.outputs[owner]?.[CHOICES_OUTPUT]) as AxisChoices | undefined
  const fresh = useFormState((s) => s.outputs[owner]?.[`${CHOICES_OUTPUT}/${axis}`]) as string[] | null | undefined
  const action = props.action ?? "axis_choices"

  const load = () => {
    setPending(true)
    const start = async () => {
      const answer = await run(action)
      if (answer === null) return false
      // The full answer is the newest: drop the per-axis lists it supersedes
      const outputs = store.getState().outputs[owner] ?? {}
      const stale = Object.keys(outputs).filter((key) => key.startsWith(`${CHOICES_OUTPUT}/`))
      if (stale.length > 0) store.getState().setOutputs(owner, Object.fromEntries(stale.map((key) => [key, undefined])))
      return true
    }
    void shared(store, `${owner}/${action}`, start)
      .then((ok) => setFailed(!ok))
      .finally(() => setPending(false))
  }

  useEffect(() => {
    // Once per schema load (its outputs are cleared then); after a failure, again on an axis change.
    if (all === undefined) load()
    // oxlint-disable-next-line react-hooks/exhaustive-deps -- `load` is recreated every render
  }, [all, action, axis])

  const list = fresh !== undefined ? fresh : all?.[axis]
  const options = list ?? EMPTY
  const selected = Array.isArray(value) ? value : EMPTY
  const loading = pending || (all === undefined && !failed)
  const broken = (all === undefined && failed) || list === null
  const status = loading ? tw("Loading values…") : broken ? tw("Could not load the values") : null

  return (
    <Combobox<string, true>
      multiple
      items={options}
      value={selected}
      onValueChange={(next) => onChange(next)}
      inputValue={query}
      onInputValueChange={setQuery}
      onOpenChange={(open) => {
        if (open && broken && !pending) load()
      }}
      disabled={disabled}
    >
      <ComboboxChips ref={anchor} className="w-full">
        <ComboboxValue>
          {(values: string[]) => (
            <>
              {values.map((item) => (
                <ComboboxChip key={item}>{item}</ComboboxChip>
              ))}
              <ComboboxChipsInput
                id={id}
                aria-label={label || tw("Axis values")}
                aria-busy={loading || undefined}
                placeholder={values.length === 0 ? (status ?? tw("Select values…")) : undefined}
              />
            </>
          )}
        </ComboboxValue>
      </ComboboxChips>
      <ComboboxContent anchor={anchor}>
        <ComboboxEmpty>{status ?? tw("No matching values")}</ComboboxEmpty>
        <ComboboxList>
          {(item: string) => (
            <ComboboxItem key={item} value={item}>
              {item}
            </ComboboxItem>
          )}
        </ComboboxList>
      </ComboboxContent>
    </Combobox>
  )
}
