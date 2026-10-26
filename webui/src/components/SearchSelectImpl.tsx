/**
 * The search select's implementation (Base UI Combobox). Import
 * `@/components/SearchSelect` instead: it renders a stand-in of the closed
 * select and loads this module when the pointer enters it, it gets focus or
 * is pressed, so a page's first load does not carry the combobox code.
 */
import { Combobox } from "@base-ui/react/combobox"
import { Check, CornerDownLeft } from "lucide-react"
import { useEffect, useLayoutEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { cn } from "@/lib/utils"
import { standInFocus } from "@/lib/lazy-module"
import { SEARCH_SELECT_TRIGGER_CLASS, selectedOption, type SearchSelectProps, type SelectOption } from "./search-select-shared"
import { SearchSelectTriggerContent } from "./SearchSelectTriggerContent"

/**
 * The trigger's accessible name as the user hears it: `aria-label`, then the
 * `aria-labelledby` targets, then a `<label for>`. The popup (a `role="dialog"`
 * because its search field lives inside it) is named after it.
 */
function triggerName(trigger: HTMLElement | null): string {
  if (!trigger) return ""
  const own = trigger.getAttribute("aria-label")?.trim()
  if (own) return own
  const ids = trigger.getAttribute("aria-labelledby")?.split(/\s+/).filter(Boolean) ?? []
  const byIds = ids
    .map((id) => trigger.ownerDocument.getElementById(id)?.textContent?.trim() ?? "")
    .filter(Boolean)
    .join(" ")
  if (byIds) return byIds
  const labels = (trigger as HTMLButtonElement).labels
  return labels?.[0]?.textContent?.trim() ?? ""
}

type ImplProps = SearchSelectProps & {
  /** Opens once mounted, the way a key press on the trigger does (the stand-in it replaces was pressed). */
  openOnMount?: boolean
  /** The stand-in this replaces: takes the focus when mounted if the stand-in had it. */
  standInKey?: string
}

export function SearchSelectImpl({ openOnMount, standInKey, ...props }: ImplProps) {
  const { t } = useTranslation()
  const { options, value, onValueChange, allowCustom = false, size = "default" } = props
  const [query, setQuery] = useState("")
  const trigger = useRef<HTMLButtonElement>(null)
  const [popupName, setPopupName] = useState("")
  useLayoutEffect(() => {
    if (standInKey && standInFocus.take(standInKey)) trigger.current?.focus()
    // oxlint-disable-next-line react/exhaustive-deps -- once, when the implementation replaces the stand-in
  }, [])
  // the stand-in was pressed: open as a key press on the trigger does (the combobox's own focus handling applies)
  useEffect(() => {
    if (!openOnMount) return
    trigger.current?.focus()
    trigger.current?.click()
    // oxlint-disable-next-line react/exhaustive-deps -- once, when the implementation replaces the stand-in
  }, [])
  const selected = selectedOption(options, value)
  const trimmed = query.trim()
  const offerCustom = allowCustom && trimmed !== "" && !options.some((o) => o.label === trimmed || o.value === trimmed)
  // An exact match goes first, so Enter after typing "Euler" picks Euler, not "Euler a".
  const exact = trimmed ? options.find((o) => o.label.toLowerCase() === trimmed.toLowerCase()) : undefined
  const items = exact ? [exact, ...options.filter((o) => o !== exact)] : options

  return (
    <Combobox.Root<SelectOption>
      items={items}
      value={selected}
      onValueChange={(next) => {
        if (next) onValueChange(next.value)
      }}
      inputValue={query}
      onInputValueChange={setQuery}
      itemToStringLabel={(o) => o.label}
      itemToStringValue={(o) => o.value}
      isItemEqualToValue={(a, b) => a.value === b.value}
      disabled={props.disabled}
      // Enter picks the first match (with custom values, Enter commits the typed text instead).
      autoHighlight={!allowCustom}
      onOpenChange={(open) => {
        if (open) setPopupName(triggerName(trigger.current))
        else setQuery("")
      }}
    >
      <Combobox.Trigger
        ref={trigger}
        id={props.id}
        // the value may be truncated in a narrow trigger: hovering shows all of it
        title={selected?.label}
        aria-label={props["aria-label"]}
        aria-labelledby={props["aria-labelledby"]}
        aria-describedby={props["aria-describedby"]}
        aria-invalid={props["aria-invalid"] || undefined}
        data-size={size}
        className={cn(SEARCH_SELECT_TRIGGER_CLASS, props.className)}
      >
        <SearchSelectTriggerContent selected={selected} placeholder={props.placeholder ?? t("Select…")} prefix={props.prefix} suffix={props.suffix} />
      </Combobox.Trigger>
      <Combobox.Portal>
        <Combobox.Positioner sideOffset={4} align="start" className="isolate z-50">
          <Combobox.Popup
            aria-label={popupName || props.placeholder || t("Options")}
            className={cn(
              "cn-menu-translucent flex max-h-[min(24rem,var(--available-height))] w-(--anchor-width) min-w-56 max-w-[min(32rem,var(--available-width))] origin-(--transform-origin) flex-col overflow-hidden rounded-lg bg-popover text-popover-foreground shadow-md ring-1 ring-foreground/10",
              "duration-100 data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95",
            )}
          >
            <div className="border-b p-1">
              <Combobox.Input
                placeholder={t("Search…")}
                className="h-8 w-full rounded-md bg-input/30 px-2 text-sm outline-none placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-ring/50"
                onKeyDown={(event) => {
                  if (event.key === "Enter" && offerCustom) {
                    event.preventDefault()
                    onValueChange(trimmed)
                    setQuery("")
                  }
                }}
              />
            </div>
            {offerCustom ? (
              <button
                type="button"
                className="mx-1 mt-1 flex items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-accent"
                onClick={() => {
                  onValueChange(trimmed)
                  setQuery("")
                }}
              >
                <CornerDownLeft className="size-3.5 text-muted-foreground" aria-hidden />
                <span className="truncate">{t("Use “{{value}}”", { value: trimmed })}</span>
              </button>
            ) : null}
            <Combobox.Empty className="px-3 py-5 text-center text-sm text-muted-foreground empty:hidden">
              {t("No matches")}
            </Combobox.Empty>
            <Combobox.List className="min-h-0 flex-1 scroll-py-1 overflow-y-auto overscroll-contain p-1 empty:hidden">
              {(option: SelectOption) => (
                <Combobox.Item
                  key={option.value}
                  value={option}
                  className="relative flex cursor-default items-center gap-2 rounded-md py-1.5 pr-8 pl-2 text-sm outline-none select-none data-highlighted:bg-accent data-highlighted:text-accent-foreground data-disabled:opacity-50"
                >
                  <span className="min-w-0 flex-1 truncate">{option.label}</span>
                  {option.hint ? <span className="shrink-0 text-xs text-muted-foreground">{option.hint}</span> : null}
                  <Combobox.ItemIndicator className="absolute right-2 flex size-4 items-center justify-center">
                    <Check className="size-4" aria-hidden />
                  </Combobox.ItemIndicator>
                </Combobox.Item>
              )}
            </Combobox.List>
          </Combobox.Popup>
        </Combobox.Positioner>
      </Combobox.Portal>
    </Combobox.Root>
  )
}
