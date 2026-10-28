/** What the search select's stand-in (`SearchSelect.tsx`) and its implementation (`SearchSelectImpl.tsx`) share (the trigger's content: `SearchSelectTriggerContent.tsx`). */
import type { ReactNode } from "react"
import { cn } from "@/lib/utils"

export type SelectOption = { value: string; label: string; hint?: string }

export type SearchSelectProps = {
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
  /** Short text after the value that never truncates (e.g. a loading stage); decorative, announce it elsewhere. */
  suffix?: ReactNode
  size?: "sm" | "default"
}

export const SEARCH_SELECT_TRIGGER_CLASS = cn(
  "flex w-full min-w-0 items-center justify-between gap-1.5 rounded-sm border border-transparent bg-field shadow-[inset_0_-1px_0_var(--input)] pr-2 pl-2.5 text-left text-sm whitespace-nowrap transition-[color,background-color,box-shadow] outline-none select-none hover:bg-field-hover",
  "focus-visible:shadow-[inset_0_-2px_0_var(--ring)] disabled:cursor-not-allowed disabled:opacity-50",
  "aria-invalid:border-destructive aria-invalid:ring-3 aria-invalid:ring-destructive/20",
  "data-[size=default]:h-8 data-[size=sm]:h-7",
)

/** The option a value shows as (a value not among the options shows as itself). */
export function selectedOption(options: readonly SelectOption[], value: string | null): SelectOption | null {
  return options.find((o) => o.value === value) ?? (value ? { value, label: value } : null)
}
