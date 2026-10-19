/**
 * A tab strip that never scrolls, wraps or cuts a label: when its tabs do
 * not fit the available width (phones, translated labels, many tabs), the
 * strip turns into a select listing the same tabs. The tab list stays
 * mounted (invisible) so it keeps naming the panels and can be measured:
 * once it fits again, the tabs come back.
 */
import { type ReactNode, useLayoutEffect, useRef, useState } from "react"
import { TabsList } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { cn } from "@/lib/utils"

export type FitTabOption<V extends string | number> = { value: V; label: string; icon?: ReactNode; disabled?: boolean }

/** Whether the strip overflows: measured on the list itself (natural width) against its slot. */
function useOverflows(slot: React.RefObject<HTMLElement | null>, list: React.RefObject<HTMLElement | null>, deps: unknown) {
  const [compact, setCompact] = useState(false)
  useLayoutEffect(() => {
    const slotEl = slot.current
    const listEl = list.current
    if (!slotEl || !listEl) return
    const update = () => {
      // in compact mode the list is `w-max` (its natural width); otherwise it is as wide as its slot
      const natural = listEl.scrollWidth
      const available = slotEl.clientWidth
      setCompact(natural > available + 1)
    }
    update()
    const observer = new ResizeObserver(update)
    observer.observe(slotEl)
    observer.observe(listEl)
    return () => observer.disconnect()
  }, [slot, list, deps])
  return compact
}

export function FitTabsList<V extends string | number>(props: {
  options: readonly FitTabOption<V>[]
  value: V | null
  onValueChange: (value: V) => void
  /** Names the tab list and, in compact mode, the select. */
  "aria-label": string
  variant?: "default" | "line"
  className?: string
  /** The tab triggers (`TabsTrigger`), one per option. */
  children: ReactNode
}) {
  const slot = useRef<HTMLDivElement>(null)
  const list = useRef<HTMLDivElement>(null)
  const compact = useOverflows(slot, list, props.options.map((o) => o.label).join("\u0000"))
  const items = props.options.map((o) => ({ value: o.value, label: o.label }))
  return (
    <div ref={slot} className="relative min-w-0">
      {/* invisible, not removed: hidden tabs still name their panels (aria-labelledby) */}
      <div className={cn(compact && "pointer-events-none invisible absolute inset-x-0 top-0 h-0 overflow-hidden")}>
        <TabsList
          ref={list}
          variant={props.variant}
          aria-label={props["aria-label"]}
          className={cn("justify-start overflow-hidden", compact ? "w-max" : "w-full", props.className)}
        >
          {props.children}
        </TabsList>
      </div>
      {compact ? (
        <Select<V> items={items} value={props.value} onValueChange={(next) => next !== null && props.onValueChange(next)}>
          <SelectTrigger aria-label={props["aria-label"]} className="w-full max-w-sm">
            <SelectValue />
          </SelectTrigger>
          <SelectContent alignItemWithTrigger={false} align="start">
            {props.options.map((option) => (
              <SelectItem key={String(option.value)} value={option.value} disabled={option.disabled}>
                {option.icon}
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      ) : null}
    </div>
  )
}
