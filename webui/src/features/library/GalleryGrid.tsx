/**
 * The gallery grid: justified rows or masonry columns at each image's exact
 * aspect ratio, virtualized (only tiles near the viewport exist), loading
 * the next page well before the end scrolls into view.
 *
 * Scrolling re-renders the grid every frame, so that work stays small: the
 * layout (aspects, justified rows) and the virtualizer's callbacks change
 * only with the items, width or layout (a new `getItemKey` makes TanStack
 * Virtual recompute every row's position), and tiles are memoised (only
 * the ones entering the range render). Tiles are size- and paint-contained
 * and their thumbnails decode off the main thread
 * (docs/research/ui-performance.md, "Scrolling").
 *
 * A tile is a button: click opens the viewer, or — once something is
 * selected, or with Ctrl / Cmd — toggles its selection; Shift extends the
 * selection. The check box and star on each tile are their own (pointer)
 * controls. The grid is one tab stop: the arrow keys move between tiles
 * (↑ / ↓ to the image above / below), Home / End to the first / last
 * loaded image, Enter opens, Space selects (Shift+Space extends), F stars.
 * The tab stop is the current tile while it is rendered, else the first
 * tile in view (after scrolling away from it); the focused tile stays
 * rendered while it has the focus, wherever the grid is scrolled.
 */
import { defaultRangeExtractor, useVirtualizer } from "@tanstack/react-virtual"
import { Check, Star } from "lucide-react"
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react"
import type { TFunction } from "i18next"
import { useTranslation } from "react-i18next"
import { api } from "@/api/endpoints"
import type { GalleryItem } from "@/api/types"
import { cn } from "@/lib/utils"
import { splitInfotext } from "@/features/infotext/display"
import type { GalleryLayout } from "./filters"
import { aspectOf, justifiedNeighbor, justifiedRows, masonryColumns, rowOf, thumbWidth, TILE_SIZES, type TileSize } from "./layout"

const GAP = 6
const PADDING = 12
/**
 * The next page is requested once fewer than this many viewport heights of
 * tiles are left below the view: at a fast flick (about 18,000 px/s) a page
 * of thumbnails arrives and is laid out before the end comes into view.
 */
const PREFETCH_VIEWPORTS = 4

type GalleryGridProps = {
  items: readonly GalleryItem[]
  layout: GalleryLayout
  size: TileSize
  selected: ReadonlySet<string>
  /** Click on a tile: `select` is set when the click changes the selection instead of opening. */
  onActivate: (item: GalleryItem, event: { shiftKey: boolean; ctrlKey: boolean; metaKey: boolean }) => void
  onToggleSelect: (item: GalleryItem, event: { shiftKey: boolean }) => void
  onStar: (item: GalleryItem) => void
  hasMore: boolean
  loadingMore: boolean
  onLoadMore: () => void
  /** Total count for screen readers (`aria-setsize`). */
  total?: number
}

function useWidth(element: HTMLElement | null): number {
  const [width, setWidth] = useState(0)
  useEffect(() => {
    if (!element) return
    const update = () => setWidth(Math.max(0, element.clientWidth - PADDING * 2))
    update()
    const observer = new ResizeObserver(update)
    observer.observe(element)
    return () => observer.disconnect()
  }, [element])
  return width
}

// Parsing an infotext per tile mount showed in fast scrolls: each item's label is computed once.
const labels = new WeakMap<GalleryItem, string>()

function label(item: GalleryItem): string {
  let text = labels.get(item)
  if (text === undefined) {
    const prompt = item.infotext ? splitInfotext(item.infotext).prompt : ""
    text = prompt ? prompt.slice(0, 140) : item.name
    labels.set(item, text)
  }
  return text
}

type TileProps = {
  item: GalleryItem
  width: number
  height: number
  selected: boolean
  selecting: boolean
  index: number
  total: number
  /** The tile that takes the grid's tab stop. */
  current: boolean
  onFocusTile: (index: number) => void
  register: (id: string, element: HTMLButtonElement | null) => void
  /** The grid's translate function (one i18n subscription for the grid, not one per tile). */
  t: TFunction
} & Pick<GalleryGridProps, "onActivate" | "onToggleSelect" | "onStar">

// Memoised: a scroll frame re-renders the grid, but only tiles whose props changed (new in the range) render.
const Tile = memo(function Tile({ item, width, height, selected, selecting, index, total, current, onFocusTile, register, t, onActivate, onToggleSelect, onStar }: TileProps) {
  const name = label(item)
  return (
    <div
      role="listitem"
      data-gallery-tile
      aria-posinset={index + 1}
      aria-setsize={total}
      className={cn(
        // fixed size: size and paint containment keep a tile's image load or hover from relaying out the grid
        "group/tile relative overflow-hidden rounded-md bg-muted outline-none [contain:strict]",
        selected && "ring-3 ring-primary ring-offset-2 ring-offset-background",
      )}
      style={{ width, height }}
    >
      <button
        ref={(element) => register(item.id, element)}
        type="button"
        tabIndex={current ? 0 : -1}
        aria-label={name}
        aria-pressed={selecting ? selected : undefined}
        aria-describedby="gallery-grid-keys"
        aria-keyshortcuts="Space F"
        onFocus={() => onFocusTile(index)}
        onClick={(event) => onActivate(item, event)}
        onKeyDown={(event) => {
          if (event.key === " " && !event.repeat) {
            event.preventDefault()
            onToggleSelect(item, event)
          } else if ((event.key === "f" || event.key === "F") && !event.ctrlKey && !event.metaKey && !event.altKey && !event.repeat) {
            event.preventDefault()
            onStar(item)
          }
        }}
        className="block size-full cursor-zoom-in outline-none focus-visible:ring-3 focus-visible:ring-ring focus-visible:ring-inset data-[selecting]:cursor-pointer"
        data-selecting={selecting || undefined}
      >
        <img
          src={api.thumbUrl(item.id, thumbWidth(width))}
          alt=""
          loading="lazy"
          decoding="async"
          draggable={false}
          className={cn("size-full object-cover transition-transform duration-200", selected ? "scale-[0.94] rounded-sm" : "group-hover/tile:scale-[1.02]")}
        />
      </button>
      <div className="pointer-events-none absolute inset-x-0 top-0 flex items-start justify-between gap-1 bg-gradient-to-b from-black/45 to-transparent p-1.5 opacity-0 transition-opacity group-focus-within/tile:opacity-100 group-hover/tile:opacity-100 data-[shown]:opacity-100" data-shown={selecting || selected || undefined}>
        <button
          type="button"
          role="checkbox"
          tabIndex={-1}
          aria-checked={selected}
          aria-label={t("Select {{name}}", { name: item.name })}
          onClick={(event) => onToggleSelect(item, event)}
          className={cn(
            "pointer-events-auto grid size-5 place-items-center rounded-full border-2 border-white/90 text-primary-foreground shadow outline-none focus-visible:ring-3 focus-visible:ring-ring",
            selected ? "border-primary bg-primary" : "bg-black/20 hover:bg-black/40",
          )}
        >
          {selected ? <Check className="size-3" strokeWidth={3} aria-hidden /> : null}
        </button>
        <button
          type="button"
          tabIndex={-1}
          aria-pressed={item.starred}
          aria-label={item.starred ? t("Unstar") : t("Star")}
          onClick={() => onStar(item)}
          className="pointer-events-auto grid size-6 place-items-center rounded-full text-white outline-none hover:bg-black/30 focus-visible:ring-3 focus-visible:ring-ring"
        >
          <Star className={cn("size-4 drop-shadow", item.starred && "fill-amber-400 text-amber-400")} aria-hidden />
        </button>
      </div>
      {item.starred && !selecting ? (
        <Star className="pointer-events-none absolute top-2 right-2 size-4 fill-amber-400 text-amber-400 drop-shadow group-hover/tile:opacity-0 group-focus-within/tile:opacity-0" aria-hidden />
      ) : null}
    </div>
  )
})

export function GalleryGrid(props: GalleryGridProps) {
  // TanStack Virtual's virtualizer is a mutable object the React Compiler must not memoise.
  "use no memo"
  const { t } = useTranslation()
  const { items, layout, size, selected, hasMore, loadingMore, onLoadMore } = props
  const [scroller, setScroller] = useState<HTMLDivElement | null>(null)
  const width = useWidth(scroller)
  const target = TILE_SIZES[size]
  // Recomputed when the items, width or layout change, not on every scroll frame.
  const aspects = useMemo(() => items.map((item) => aspectOf(item.width, item.height)), [items])
  const rows = useMemo(() => (layout === "justified" ? justifiedRows(aspects, width, target, GAP) : []), [layout, aspects, width, target])
  const columns = useMemo(() => masonryColumns(width, target, GAP), [width, target])
  const selecting = selected.size > 0
  const total = props.total ?? items.length

  // Roving focus: one tile is the tab stop; the arrow keys move it (scrolling the tile into view first).
  const [active, setActive] = useState(0)
  const current = Math.max(0, Math.min(active, items.length - 1))
  // While a tile has the focus its row stays rendered, so scrolling never drops the focus to the page.
  const [focusInside, setFocusInside] = useState(false)
  const pinned = focusInside && items.length > 0 ? (layout === "justified" ? rowOf(rows, current) : current) : -1

  // Stable between scroll frames: TanStack Virtual recomputes every position when `getItemKey` changes.
  const getScrollElement = useCallback(() => scroller, [scroller])
  const estimateSize = useCallback(
    (index: number) => (layout === "justified" ? (rows[index]?.height ?? target) : columns.width / (aspects[index] ?? 1)) + GAP,
    [layout, rows, target, columns, aspects],
  )
  const getItemKey = useCallback((index: number) => (layout === "justified" ? `row-${rows[index]?.start ?? index}` : (items[index]?.id ?? index)), [layout, rows, items])
  const rangeExtractor = useCallback(
    (range: Parameters<typeof defaultRangeExtractor>[0]) => {
      const indexes = defaultRangeExtractor(range)
      return pinned >= 0 && !indexes.includes(pinned) ? [...indexes, pinned].sort((a, b) => a - b) : indexes
    },
    [pinned],
  )
  // oxlint-disable-next-line react/incompatible-library -- opted out of memoisation above
  const virtualizer = useVirtualizer({
    count: layout === "justified" ? rows.length : items.length,
    getScrollElement,
    estimateSize,
    lanes: layout === "masonry" ? columns.count : 1,
    overscan: layout === "justified" ? 4 : 12,
    paddingStart: PADDING,
    paddingEnd: PADDING,
    getItemKey,
    rangeExtractor,
  })
  // Sizes depend on the width, layout and size: re-measure when they change.
  const signature = `${layout}:${size}:${Math.round(width)}:${columns.count}`
  const measured = useRef(signature)
  useEffect(() => {
    if (measured.current === signature) return
    measured.current = signature
    virtualizer.measure()
  }, [signature, virtualizer])

  const virtualItems = virtualizer.getVirtualItems()
  // Fetch ahead: the next page is requested while several screens of tiles are still left below.
  const viewport = scroller?.clientHeight ?? 0
  const left = virtualizer.getTotalSize() - (virtualizer.scrollOffset ?? 0) - viewport
  const nearEnd = items.length > 0 && left < Math.max(viewport, 1) * PREFETCH_VIEWPORTS
  useEffect(() => {
    if (hasMore && !loadingMore && nearEnd) onLoadMore()
  }, [nearEnd, items.length, hasMore, loadingMore, onLoadMore])

  // The tab stop: the current tile when rendered, else the first tile in view.
  const tileIndexes = (virtual: { index: number }) => {
    if (layout !== "justified") return [virtual.index]
    const row = rows[virtual.index]
    return row ? Array.from({ length: row.end - row.start }, (_, offset) => row.start + offset) : []
  }
  const scrollTop = virtualizer.scrollOffset ?? 0
  const inView = virtualItems.filter((virtual) => virtual.end > scrollTop)
  const rendered = virtualItems.some((virtual) => tileIndexes(virtual).includes(current))
  const tabStop = rendered ? current : (tileIndexes(inView[0] ?? virtualItems[0] ?? { index: -1 })[0] ?? current)
  const buttons = useRef(new Map<string, HTMLButtonElement>())
  const pendingFocus = useRef<string | null>(null)
  // stable (refs only), so the memoised tiles do not re-render for it
  const register = useCallback((id: string, element: HTMLButtonElement | null) => {
    if (element) {
      buttons.current.set(id, element)
      if (pendingFocus.current === id) {
        pendingFocus.current = null
        element.focus({ preventScroll: true })
      }
    } else buttons.current.delete(id)
  }, [])
  const focusIndex = (index: number) => {
    const target = Math.max(0, Math.min(index, items.length - 1))
    const item = items[target]
    if (!item) return
    setActive(target)
    virtualizer.scrollToIndex(layout === "justified" ? Math.max(0, rowOf(rows, target)) : target, { align: "auto" })
    const element = buttons.current.get(item.id)
    if (element) element.focus({ preventScroll: true })
    else pendingFocus.current = item.id
  }
  const sameLane = (index: number, direction: -1 | 1): number | null => {
    const lane = virtualizer.measurementsCache[index]?.lane
    if (lane === undefined) return null
    for (let other = index + direction; other >= 0 && other < items.length; other += direction) {
      if (virtualizer.measurementsCache[other]?.lane === lane) return other
    }
    return null
  }
  const onGridKey = (event: React.KeyboardEvent) => {
    if (event.altKey || event.ctrlKey || event.metaKey || !(event.target instanceof HTMLElement) || !event.target.closest("[data-gallery-tile]")) return
    let next: number | null = null
    if (event.key === "ArrowLeft") next = current - 1
    else if (event.key === "ArrowRight") next = current + 1
    else if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      const direction = event.key === "ArrowUp" ? -1 : 1
      next = layout === "justified" ? justifiedNeighbor(rows, current, direction, GAP) : sameLane(current, direction)
    } else if (event.key === "Home") next = 0
    else if (event.key === "End") next = items.length - 1
    else return
    event.preventDefault()
    if (next !== null && next >= 0 && next < items.length && next !== current) focusIndex(next)
  }

  const tileProps = { onActivate: props.onActivate, onToggleSelect: props.onToggleSelect, onStar: props.onStar, selecting, total, onFocusTile: setActive, register, t }

  return (
    <div ref={setScroller} className="h-full overflow-y-auto overscroll-contain" data-testid="gallery-scroller">
      <p id="gallery-grid-keys" className="sr-only">
        {t("Arrow keys move between images, Enter opens one, Space selects it, Shift+Space extends the selection, F stars it.")}
      </p>
      <div
        role="list"
        aria-label={t("Images")}
        aria-busy={loadingMore || undefined}
        className="relative"
        style={{ height: virtualizer.getTotalSize(), marginInline: PADDING }}
        onKeyDown={onGridKey}
        onFocus={() => setFocusInside(true)}
        onBlur={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setFocusInside(false)
        }}
      >
        {virtualItems.map((virtual) => {
          if (layout === "justified") {
            const row = rows[virtual.index]
            if (!row) return null
            return (
              <div key={virtual.key} role="none" className="absolute inset-x-0 top-0 flex" style={{ transform: `translateY(${virtual.start}px)`, gap: GAP }}>
                {items.slice(row.start, row.end).map((item, offset) => (
                  <Tile
                    key={item.id}
                    item={item}
                    index={row.start + offset}
                    width={row.widths[offset] ?? target}
                    height={row.height}
                    selected={selected.has(item.id)}
                    current={row.start + offset === tabStop}
                    {...tileProps}
                  />
                ))}
              </div>
            )
          }
          const item = items[virtual.index]
          if (!item) return null
          return (
            <div
              key={virtual.key}
              role="none"
              className="absolute top-0"
              style={{ transform: `translate(${virtual.lane * (columns.width + GAP)}px, ${virtual.start}px)`, width: columns.width }}
            >
              <Tile
                item={item}
                index={virtual.index}
                width={columns.width}
                height={columns.width / (aspects[virtual.index] ?? 1)}
                selected={selected.has(item.id)}
                current={virtual.index === tabStop}
                {...tileProps}
              />
            </div>
          )
        })}
      </div>
      {loadingMore ? <p className="py-3 text-center text-xs text-muted-foreground">{t("Loading…")}</p> : null}
    </div>
  )
}
