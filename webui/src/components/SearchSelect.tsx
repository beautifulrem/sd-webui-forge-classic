/**
 * A select whose popup has a search field: the right control for long lists
 * (checkpoints, samplers with aliases, extension-provided choices).
 * With `allowCustom`, the typed text can be committed as a value.
 */
import { Combobox } from "@base-ui/react/combobox"
import { Check, ChevronsUpDown, CornerDownLeft } from "lucide-react"
import { useState, type ReactNode } from "react"
import { useTranslation } from "react-i18next"
import { cn } from "@/lib/utils"

export type SelectOption = { value: string; label: string; hint?: string }

type SearchSelectProps = {
  options: readonly SelectOption[]
  value: string | null
  onValueChange: (value: string) => void
  placeholder?: string
  allowCustom?: boolean
  disabled?: boolean
  id?: string
  className?: string
  "aria-label"?: string
  "aria-labelledby"?: string
  "aria-describedby"?: string
  "aria-invalid"?: boolean
  /** Extra content in the trigger before the value (e.g. an icon). */
  prefix?: ReactNode
  size?: "sm" | "default"
}

export function SearchSelect(props: SearchSelectProps) {
  const { t } = useTranslation()
  const { options, value, onValueChange, allowCustom = false, size = "default" } = props
  const [query, setQuery] = useState("")
  const selected = options.find((o) => o.value === value) ?? (value ? { value, label: value } : null)
  const trimmed = query.trim()
  const offerCustom = allowCustom && trimmed !== "" && !options.some((o) => o.label === trimmed || o.value === trimmed)

  return (
    <Combobox.Root<SelectOption>
      items={options}
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
      onOpenChange={(open) => {
        if (!open) setQuery("")
      }}
    >
      <Combobox.Trigger
        id={props.id}
        aria-label={props["aria-label"]}
        aria-labelledby={props["aria-labelledby"]}
        aria-describedby={props["aria-describedby"]}
        aria-invalid={props["aria-invalid"] || undefined}
        data-size={size}
        className={cn(
          "flex w-full min-w-0 items-center justify-between gap-1.5 rounded-lg border border-input bg-transparent pr-2 pl-2.5 text-left text-sm whitespace-nowrap transition-colors outline-none select-none",
          "focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50",
          "aria-invalid:border-destructive aria-invalid:ring-3 aria-invalid:ring-destructive/20",
          "data-[size=default]:h-8 data-[size=sm]:h-7 dark:bg-input/30 dark:hover:bg-input/50",
          props.className,
        )}
      >
        {props.prefix}
        <span className={cn("min-w-0 flex-1 truncate", !selected && "text-muted-foreground")}>
          {selected ? selected.label : (props.placeholder ?? t("Select…"))}
        </span>
        <ChevronsUpDown className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
      </Combobox.Trigger>
      <Combobox.Portal>
        <Combobox.Positioner sideOffset={4} align="start" className="isolate z-50">
          <Combobox.Popup
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
