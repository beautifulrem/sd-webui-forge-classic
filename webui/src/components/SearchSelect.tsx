/**
 * A select whose popup has a search field: the right control for long lists
 * (checkpoints, samplers with aliases, extension-provided choices).
 * With `allowCustom`, the typed text can be committed as a value.
 *
 * The combobox code (`SearchSelectImpl.tsx`, Base UI Combobox) loads on
 * demand: until the select is used a stand-in of it renders (the same
 * trigger, a `combobox` button with the selected label). The pointer
 * entering it or focus fetches the code; a press (or an opening key) puts
 * the real select in its place and opens it (as a key press on its trigger
 * does), and keyboard focus puts it there with the focus. A stand-in is not replaced under a pointer pressing it, so a
 * click is never lost (docs/research/ui-performance.md, "Bundle").
 */
import { useId, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { cn } from "@/lib/utils"
import { lazyModule, prefetch, standInFocus } from "@/lib/lazy-module"
import { SEARCH_SELECT_TRIGGER_CLASS, selectedOption, type SearchSelectProps } from "./search-select-shared"
import { SearchSelectTriggerContent } from "./SearchSelectTriggerContent"

export type { SelectOption } from "./search-select-shared"

const implementation = lazyModule(() => import("./SearchSelectImpl"))

export function SearchSelect(props: SearchSelectProps) {
  const { t } = useTranslation()
  const module = implementation.useModule()
  const key = useId()
  const [activation, setActivation] = useState<"open" | "focus" | null>(null)
  // a press focuses the button before its click: that focus must not replace the button under the pointer
  const pressing = useRef(false)
  if (module && activation) return <module.SearchSelectImpl {...props} openOnMount={activation === "open"} standInKey={key} />

  const activate = (how: "open" | "focus") => {
    setActivation((current) => (current === "open" ? current : how))
    prefetch(implementation)
  }
  const selected = selectedOption(props.options, props.value)
  return (
    <button
      type="button"
      role="combobox"
      aria-expanded={false}
      aria-haspopup="dialog"
      id={props.id}
      title={selected?.label}
      aria-label={props["aria-label"]}
      aria-labelledby={props["aria-labelledby"]}
      aria-describedby={props["aria-describedby"]}
      aria-invalid={props["aria-invalid"] || undefined}
      disabled={props.disabled}
      data-disabled={props.disabled ? "" : undefined}
      data-size={props.size ?? "default"}
      className={cn(SEARCH_SELECT_TRIGGER_CLASS, props.className)}
      onPointerEnter={() => prefetch(implementation)}
      onPointerDown={() => {
        pressing.current = true
      }}
      onFocus={() => {
        standInFocus.focused(key)
        if (pressing.current) prefetch(implementation)
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
      onKeyDown={(event) => {
        if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return
        event.preventDefault()
        activate("open")
      }}
    >
      <SearchSelectTriggerContent selected={selected} placeholder={props.placeholder ?? t("Select…")} prefix={props.prefix} suffix={props.suffix} />
    </button>
  )
}
