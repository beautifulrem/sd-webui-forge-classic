/**
 * This session's results as a virtualized thumbnail grid (or a horizontal
 * strip on phones). One listbox tab stop: arrow keys move the selection,
 * Enter opens the viewer, Delete removes the result from the history.
 */
import { useVirtualizer } from "@tanstack/react-virtual"
import { History } from "lucide-react"
import { useEffect, useId, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { api } from "@/api/endpoints"
import { cn } from "@/lib/utils"
import { selectedItem, useSession } from "./session-store"

type HistoryGridProps = {
  onOpen: (key: string) => void
  orientation?: "grid" | "strip"
  className?: string
}

const CELL_MIN_PX = 104
const STRIP_CELL_PX = 76
const GAP_PX = 6

function useWidth(element: HTMLElement | null): number {
  const [width, setWidth] = useState(0)
  useEffect(() => {
    if (!element) return
    const update = () => setWidth(element.clientWidth)
    update()
    const observer = new ResizeObserver(update)
    observer.observe(element)
    return () => observer.disconnect()
  }, [element])
  return width
}

export function HistoryGrid({ onOpen, orientation = "grid", className }: HistoryGridProps) {
  // TanStack Virtual's virtualizer is a mutable object the React Compiler must not memoise.
  "use no memo"
  const { t } = useTranslation()
  const id = useId()
  const items = useSession((s) => s.items)
  const current = useSession((s) => selectedItem(s)?.key ?? null)
  const [scroller, setScroller] = useState<HTMLDivElement | null>(null)
  const width = useWidth(scroller)
  const strip = orientation === "strip"
  const columns = strip ? 1 : Math.max(2, Math.floor((width + GAP_PX) / (CELL_MIN_PX + GAP_PX)))
  const cell = strip ? STRIP_CELL_PX : Math.max(48, (width - GAP_PX * (columns - 1)) / columns)
  const rows = strip ? items.length : Math.ceil(items.length / columns)

  // oxlint-disable-next-line react/incompatible-library -- opted out of memoisation above
  const virtualizer = useVirtualizer({
    count: rows,
    getScrollElement: () => scroller,
    estimateSize: () => cell + GAP_PX,
    horizontal: strip,
    overscan: 4,
  })
  // Cell size follows the pane width.
  const measured = useRef(cell)
  useEffect(() => {
    if (measured.current !== cell) {
      measured.current = cell
      virtualizer.measure()
    }
  }, [cell, virtualizer])

  const currentIndex = items.findIndex((item) => item.key === current)
  const optionId = (key: string) => `${id}-${key.replace(/[^\w-]/g, "_")}`
  const virtualRows = virtualizer.getVirtualItems()
  const currentRow = currentIndex === -1 ? -1 : strip ? currentIndex : Math.floor(currentIndex / columns)
  // Only point at an option that is rendered (scrolling can virtualize it away).
  const activeDescendant = current && virtualRows.some((row) => row.index === currentRow) ? optionId(current) : undefined

  const selectIndex = (index: number) => {
    const item = items[Math.max(0, Math.min(items.length - 1, index))]
    if (!item) return
    useSession.getState().select(item.key)
    useSession.getState().setFollow(false)
    virtualizer.scrollToIndex(strip ? items.indexOf(item) : Math.floor(items.indexOf(item) / columns), { align: "auto" })
  }

  const onKeyDown = (event: React.KeyboardEvent) => {
    const at = currentIndex === -1 ? 0 : currentIndex
    const step = strip ? 1 : columns
    const keys: Record<string, () => void> = {
      ArrowLeft: () => selectIndex(at - 1),
      ArrowRight: () => selectIndex(at + 1),
      ArrowUp: () => selectIndex(strip ? at - 1 : at - step),
      ArrowDown: () => selectIndex(strip ? at + 1 : at + step),
      Home: () => selectIndex(0),
      End: () => selectIndex(items.length - 1),
      Enter: () => current && onOpen(current),
      " ": () => current && onOpen(current),
      Delete: () => current && useSession.getState().remove(current),
      Backspace: () => current && useSession.getState().remove(current),
    }
    const run = keys[event.key]
    if (run && !event.altKey && !event.ctrlKey && !event.metaKey) {
      event.preventDefault()
      run()
    }
  }

  if (items.length === 0) {
    return (
      <div className={cn("flex flex-col items-center justify-center gap-2 p-6 text-center text-xs text-muted-foreground", className)}>
        <History className="size-6 opacity-50" aria-hidden />
        <p>{t("Results of this session appear here")}</p>
      </div>
    )
  }

  return (
    <div
      ref={setScroller}
      role="listbox"
      aria-label={t("History")}
      aria-orientation={strip ? "horizontal" : "vertical"}
      aria-activedescendant={activeDescendant}
      tabIndex={0}
      onKeyDown={onKeyDown}
      className={cn(
        "relative min-h-0 overscroll-contain rounded-lg outline-none focus-visible:ring-3 focus-visible:ring-ring/50",
        strip ? "overflow-x-auto overflow-y-hidden" : "overflow-y-auto",
        className,
      )}
    >
      <div
        className="relative"
        style={strip ? { width: virtualizer.getTotalSize(), height: STRIP_CELL_PX } : { height: virtualizer.getTotalSize(), width: "100%" }}
      >
        {virtualRows.map((row) => {
          const rowItems = strip ? [items[row.index]!] : items.slice(row.index * columns, row.index * columns + columns)
          return (
            <div
              key={row.key}
              className="absolute top-0 left-0 flex"
              style={
                strip
                  ? { transform: `translateX(${row.start}px)`, height: STRIP_CELL_PX }
                  : { transform: `translateY(${row.start}px)`, width: "100%", gap: GAP_PX }
              }
            >
              {rowItems.map((item) => {
                const selected = item.key === current
                return (
                  <div
                    key={item.key}
                    id={optionId(item.key)}
                    role="option"
                    aria-selected={selected}
                    aria-label={t("Result {{index}}, {{tab}}", { index: items.indexOf(item) + 1, tab: item.tab })}
                    onClick={() => {
                      useSession.getState().select(item.key)
                      useSession.getState().setFollow(false)
                    }}
                    onDoubleClick={() => onOpen(item.key)}
                    className={cn(
                      "group/cell relative shrink-0 cursor-pointer overflow-hidden rounded-md bg-checker ring-offset-1 ring-offset-background transition-shadow",
                      selected ? "ring-2 ring-primary" : "hover:ring-2 hover:ring-ring/40",
                    )}
                    style={{ width: cell, height: cell, marginRight: strip ? GAP_PX : undefined }}
                  >
                    <img
                      src={api.thumbUrl(item.fileId, 256)}
                      alt=""
                      loading="lazy"
                      decoding="async"
                      draggable={false}
                      className="size-full object-cover"
                    />
                    {item.tab === "img2img" ? (
                      <span className="absolute bottom-1 left-1 rounded bg-black/55 px-1 text-[0.6rem] font-medium text-white">i2i</span>
                    ) : null}
                  </div>
                )
              })}
            </div>
          )
        })}
      </div>
    </div>
  )
}
