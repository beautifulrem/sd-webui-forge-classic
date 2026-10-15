/** Shared pieces of the prompt tools panels. */
import { Search, X } from "lucide-react"
import type { ReactNode } from "react"
import { useTranslation } from "react-i18next"
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group"
import { cn } from "@/lib/utils"

export function PanelHeader({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="flex items-center gap-2 border-b px-3 py-2">
      <h2 className="truncate text-sm font-semibold">{title}</h2>
      <div className="ml-auto flex shrink-0 items-center gap-1">{children}</div>
    </div>
  )
}

export function SearchBox({ value, onChange, placeholder, className }: { value: string; onChange: (value: string) => void; placeholder: string; className?: string }) {
  const { t } = useTranslation()
  return (
    <InputGroup className={cn("h-8", className)}>
      <InputGroupAddon>
        <Search aria-hidden />
      </InputGroupAddon>
      <InputGroupInput type="search" value={value} placeholder={placeholder} aria-label={placeholder} onChange={(event) => onChange(event.target.value)} />
      {value ? (
        <InputGroupAddon align="inline-end">
          <InputGroupButton size="icon-xs" aria-label={t("Clear the search")} onClick={() => onChange("")}>
            <X />
          </InputGroupButton>
        </InputGroupAddon>
      ) : null}
    </InputGroup>
  )
}
