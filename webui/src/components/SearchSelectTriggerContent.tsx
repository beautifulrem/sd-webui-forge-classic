import { ChevronsUpDown } from "lucide-react"
import type { ReactNode } from "react"
import { cn } from "@/lib/utils"
import type { SelectOption } from "./search-select-shared"

/** The closed select's content: the prefix, the selected label (or the placeholder), the suffix and the chevron. */
export function SearchSelectTriggerContent({ selected, placeholder, prefix, suffix }: { selected: SelectOption | null; placeholder: string; prefix?: ReactNode; suffix?: ReactNode }) {
  return (
    <>
      {prefix}
      <span className={cn("min-w-0 flex-1 truncate", !selected && "text-muted-foreground")}>{selected ? selected.label : placeholder}</span>
      {suffix ? (
        <span aria-hidden className="max-w-[60%] shrink-0 truncate text-xs text-muted-foreground tabular-nums">
          {suffix}
        </span>
      ) : null}
      <ChevronsUpDown className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
    </>
  )
}
