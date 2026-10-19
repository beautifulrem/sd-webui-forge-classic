/**
 * The virtualized card grid / list. Only visible rows (plus a small overscan,
 * capped at MAX_MOUNTED cards) are in the DOM; it reports the item range it
 * shows (the browser loads the windows covering it) and shows skeletons for
 * items not loaded yet. One listbox tab stop: arrows,
 * Page Up/Down, Home/End move the cursor, Enter/Space apply the card, I opens
 * the metadata, E the editor, Shift+F10 / the Menu key / "." the item menu.
 */
import { useVirtualizer } from "@tanstack/react-virtual"
import { type Ref, useEffect, useId, useImperativeHandle, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import type { NetworkItem, NetworkKind } from "@/api/types"
import { Skeleton } from "@/components/ui/skeleton"
import { cn } from "@/lib/utils"
import { GAP, gridLayout, moveIndex } from "./grid-layout"
import { isEditable, type MenuAnchor } from "./items"
import { type ItemAction, NetworkCard, NetworkRow } from "./NetworkCard"
import { type IndexRange, itemKey } from "./queries"
import type { CardSize, ViewMode } from "./store"

export type GridHandle = { focus: () => void }

type GridProps = {
  /** Items by index; `undefined` while its window loads. */
  items: readonly (NetworkItem | undefined)[]
  total: number
  kind: NetworkKind | undefined
  view: ViewMode
  size: CardSize
  isActive: (item: NetworkItem) => boolean
  currentArch: string | null | undefined
  /** The item indices in view (with the overscan) changed. */
  onRangeChange: (range: IndexRange) => void
  onApply: (item: NetworkItem) => void
  onAction: (action: ItemAction, item: NetworkItem, anchor: MenuAnchor) => void
  label: string
  className?: string
  ref?: Ref<GridHandle>
}

function useSize(element: HTMLElement | null) {
  const [size, setSize] = useState({ width: 0, height: 0 })
  useEffect(() => {
    if (!element) return
    const update = () => setSize((old) => (old.width === element.clientWidth && old.height === element.clientHeight ? old : { width: element.clientWidth, height: element.clientHeight }))
    update()
    const observer = new ResizeObserver(update)
    observer.observe(element)
    return () => observer.disconnect()
  }, [element])
  return size
}

const mismatchOf = (item: NetworkItem, kind: NetworkKind | undefined, current: string | null | undefined) =>
  Boolean(item.arch && current && item.arch !== current && (kind?.action === "prompt" || kind?.action === "checkpoint"))

export function NetworkGrid({ items, total, kind, view, size, isActive, currentArch, onRangeChange, onApply, onAction, label, className, ref }: GridProps) {
  // TanStack Virtual's virtualizer is a mutable object the React Compiler must not memoise.
  "use no memo"
  const { t } = useTranslation()
  const id = useId()
  const [scroller, setScroller] = useState<HTMLDivElement | null>(null)
  const { width, height } = useSize(scroller)
  // padding (p-2 = 2 × 0.5rem) is outside the cells
  const inner = Math.max(0, width - 16)
  const layout = gridLayout(inner, height, view, size)
  const count = Math.max(total, items.length)
  const rows = Math.ceil(count / layout.columns)
  const [cursorState, setCursor] = useState(-1)
  // A shorter list (new search) keeps the cursor inside it.
  const cursor = Math.min(cursorState, count - 1)
  const cursorRef = useRef(cursor)
  cursorRef.current = cursor

  useImperativeHandle(ref, () => ({ focus: () => scroller?.focus() }), [scroller])

  // oxlint-disable-next-line react/incompatible-library -- opted out of memoisation above
  const virtualizer = useVirtualizer({
    count: rows,
    getScrollElement: () => scroller,
    estimateSize: () => layout.rowHeight,
    overscan: layout.overscan,
    paddingStart: 8,
    paddingEnd: 8,
  })
  const pitch = useRef(layout.rowHeight)
  useEffect(() => {
    if (pitch.current !== layout.rowHeight) {
      pitch.current = layout.rowHeight
      virtualizer.measure()
    }
  }, [layout.rowHeight, virtualizer])

  const virtualRows = virtualizer.getVirtualItems()
  const firstIndex = (virtualRows[0]?.index ?? 0) * layout.columns
  const lastIndex = Math.min(count, ((virtualRows.at(-1)?.index ?? -1) + 1) * layout.columns) - 1
  const reportRange = useRef(onRangeChange)
  reportRange.current = onRangeChange
  useEffect(() => {
    reportRange.current({ start: firstIndex, end: lastIndex })
  }, [firstIndex, lastIndex])
  let waiting = false
  for (let index = firstIndex; index <= lastIndex && !waiting; index++) waiting = !items[index]

  const optionId = (index: number) => `${id}-o${index}`
  const cursorRow = cursor < 0 ? -1 : Math.floor(cursor / layout.columns)
  const cursorItem = cursor >= 0 ? items[cursor] : undefined
  const activeDescendant = cursorItem && virtualRows.some((row) => row.index === cursorRow) ? optionId(cursor) : undefined

  const moveTo = (index: number) => {
    setCursor(index)
    virtualizer.scrollToIndex(Math.floor(index / layout.columns), { align: "auto" })
  }

  const elementOf = (index: number) => scroller?.querySelector<HTMLElement>(`[data-index="${index}"]`) ?? scroller

  const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.target !== event.currentTarget || event.ctrlKey || event.metaKey || event.altKey) return
    const pageRows = Math.max(1, Math.floor(height / layout.rowHeight) - 1)
    const next = moveIndex(event.key, cursorRef.current, count, layout.columns, pageRows)
    if (next !== null) {
      event.preventDefault()
      moveTo(next)
      return
    }
    const item = cursorItem
    if (!item) return
    const anchor = elementOf(cursor)
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault()
      onApply(item)
    } else if ((event.key === "F10" && event.shiftKey) || event.key === "ContextMenu" || event.key === ".") {
      event.preventDefault()
      if (anchor) onAction("menu", item, anchor)
    } else if (event.key.toLowerCase() === "i" && !event.shiftKey) {
      event.preventDefault()
      if (anchor) onAction("metadata", item, anchor)
    } else if (event.key.toLowerCase() === "e" && !event.shiftKey && (kind?.fields.length ?? 0) > 0 && isEditable(item)) {
      event.preventDefault()
      if (anchor) onAction("edit", item, anchor)
    }
  }

  // Stable across renders (this grid renders on every scroll frame): the memoised cards keep
  // their props and are skipped; the newest handlers are read through a ref.
  const handlers = useRef({ onApply, onAction })
  handlers.current = { onApply, onAction }
  const [{ apply, action }] = useState(() => ({
    apply: (item: NetworkItem, index: number) => {
      setCursor(index)
      handlers.current.onApply(item)
    },
    action: (name: ItemAction, item: NetworkItem, index: number, anchor: MenuAnchor) => {
      setCursor(index)
      handlers.current.onAction(name, item, anchor)
    },
  }))

  const Item = view === "list" ? NetworkRow : NetworkCard
  return (
    <>
      <div
        ref={setScroller}
        role="listbox"
        aria-label={label}
        aria-multiselectable="true"
        aria-activedescendant={activeDescendant}
        aria-busy={waiting || undefined}
        aria-keyshortcuts="Enter I E Shift+F10"
        aria-orientation={view === "list" ? "vertical" : undefined}
        tabIndex={0}
        onKeyDown={onKeyDown}
        onFocus={(event) => {
          if (event.target === event.currentTarget && cursorRef.current < 0 && items.length > 0) setCursor(0)
        }}
        className={cn("@container relative min-h-0 overflow-y-auto overscroll-contain px-2 outline-none focus-visible:ring-3 focus-visible:ring-ring/40 focus-visible:ring-inset", className)}
      >
        <div className="relative w-full" style={{ height: virtualizer.getTotalSize() }}>
          {virtualRows.map((row) => {
            const start = row.index * layout.columns
            const cells = []
            for (let column = 0; column < layout.columns; column++) {
              const index = start + column
              if (index >= count) break
              const item = items[index]
              cells.push(
                <div key={item ? itemKey(item) : `pending-${index}`} role="none" style={{ width: layout.cellWidth, height: layout.cellHeight }} className="shrink-0">
                  {item ? (
                    <Item
                      item={item}
                      kind={kind}
                      id={optionId(index)}
                      index={index}
                      active={isActive(item)}
                      cursor={index === cursor}
                      mismatch={mismatchOf(item, kind, currentArch)}
                      width={layout.cellWidth}
                      onApply={apply}
                      onAction={action}
                    />
                  ) : (
                    <Skeleton aria-hidden className={view === "list" ? "size-full rounded-lg" : "size-full rounded-xl"} />
                  )}
                </div>,
              )
            }
            return (
              <div
                key={row.key}
                role="none"
                className="absolute top-0 left-0 flex w-full"
                style={{ transform: `translateY(${row.start}px)`, gap: GAP, height: layout.cellHeight }}
              >
                {cells}
              </div>
            )
          })}
        </div>
      </div>
      {/* outside the listbox: a listbox may only contain options */}
      {count === 0 ? null : (
        <p className="sr-only" aria-live="polite">
          {t("Items: {{count}}", { count })}
        </p>
      )}
    </>
  )
}
