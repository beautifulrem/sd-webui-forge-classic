"use client"

/**
 * Resizable panels: a row (or column) of panels with draggable separators,
 * the subset of react-resizable-panels (v4) the app uses, with its API and
 * semantics: sizes are pixels when numbers and percentages when strings
 * ("34"), panels can be collapsible (and collapsed through a panel ref),
 * a separator is a focusable `separator` that the arrow keys move by 5 %
 * (Home / End to the limits, Enter collapses or expands the panel before it,
 * F6 moves to the next separator), and `useDefaultLayout` keeps the layout in
 * `localStorage` under the library's key and format. The library itself
 * weighed 14.5 KB gzip in the Generate page's first load
 * (docs/research/ui-performance.md, "Bundle").
 *
 * A panel's element carries its `id` and its share as `flex-grow` (a
 * percentage of the group); the separator before a panel controls it.
 */
import * as React from "react"
import { cn } from "cn"

type Size = number | string
type Orientation = "horizontal" | "vertical"
/** Panel id → percentage of the group. */
export type Layout = Record<string, number>
type PanelSize = { asPercentage: number; inPixels: number }

export type PanelImperativeHandle = {
  collapse: () => void
  expand: () => void
  isCollapsed: () => boolean
  getSize: () => PanelSize
  resize: (size: Size) => void
}

type PanelConstraints = {
  id: string
  defaultSize?: Size
  minSize?: Size
  maxSize?: Size
  collapsible: boolean
  collapsedSize: Size
  preservePixels: boolean
  onResize?: (size: PanelSize, id: string) => void
}

type GroupContextValue = {
  orientation: Orientation
  layout: Layout | null
  register: (panel: PanelConstraints) => () => void
  panelHandle: (id: string) => PanelImperativeHandle
  separatorProps: (element: HTMLElement | null) => SeparatorState | null
  startDrag: (element: HTMLElement, event: React.PointerEvent) => void
  keyResize: (element: HTMLElement, event: React.KeyboardEvent) => void
  resetSeparator: (element: HTMLElement) => void
  /** Re-renders separators once the panel order is known. */
  version: number
}

type SeparatorState = { controls: string; now: number; min: number; max: number }

const GroupContext = React.createContext<GroupContextValue | null>(null)

const round = (value: number) => Math.round(value * 1000) / 1000

/** A size in percent of a group `axis` pixels long. */
function percent(size: Size | undefined, axis: number, fallback: number): number {
  if (size === undefined) return fallback
  if (typeof size === "number") return axis > 0 ? (size / axis) * 100 : fallback
  const value = Number.parseFloat(size)
  return Number.isFinite(value) ? value : fallback
}

type Bounds = { min: number; max: number; collapsed: number; collapsible: boolean }

function bounds(panel: PanelConstraints, axis: number): Bounds {
  const min = Math.min(100, percent(panel.minSize, axis, 0))
  return { min, max: Math.max(min, percent(panel.maxSize, axis, 100)), collapsed: percent(panel.collapsedSize, axis, 0), collapsible: panel.collapsible }
}

function isCollapsedSize(size: number, b: Bounds): boolean {
  return b.collapsible && Math.abs(size - b.collapsed) < 0.01
}

/** Clamps one panel's size to its constraints; below the minimum a collapsible panel collapses past halfway. */
function clampSize(size: number, b: Bounds): number {
  if (b.collapsible && size < b.min) return size < (b.min + b.collapsed) / 2 ? b.collapsed : b.min
  return Math.min(b.max, Math.max(b.min, size))
}

/**
 * Brings a layout within the constraints and to a total of 100 %: each panel
 * is clamped, then the difference goes to the panels that can take it, those
 * without a size of their own (the flexible ones) first.
 */
function validate(sizes: number[], panels: PanelConstraints[], axis: number): number[] {
  const all = panels.map((panel) => bounds(panel, axis))
  const next = sizes.map((size, index) => (isCollapsedSize(size, all[index]!) ? all[index]!.collapsed : clampSize(size, all[index]!)))
  let diff = 100 - next.reduce((sum, size) => sum + size, 0)
  const order = panels
    .map((panel, index) => ({ index, flexible: panel.defaultSize === undefined && !panel.preservePixels }))
    .sort((a, b) => Number(b.flexible) - Number(a.flexible))
    .map(({ index }) => index)
  for (const index of order) {
    if (Math.abs(diff) < 0.001) break
    const b = all[index]!
    if (isCollapsedSize(next[index]!, b) && b.collapsed < b.min) continue
    const target = Math.min(b.max, Math.max(b.min, next[index]! + diff))
    diff -= target - next[index]!
    next[index] = target
  }
  return next.map(round)
}

/**
 * Moves the boundary after panel `index` by `delta` percent (from `start`): the two panels around
 * it change, each within its constraints (collapsing when dragged far enough below its minimum).
 */
function moveBoundary(start: number[], index: number, delta: number, panels: PanelConstraints[], axis: number): number[] {
  const a = bounds(panels[index]!, axis)
  const b = bounds(panels[index + 1]!, axis)
  const total = start[index]! + start[index + 1]!
  let first = clampSize(start[index]! + delta, a)
  let second = total - first
  const clampedSecond = clampSize(second, b)
  if (clampedSecond !== second) {
    second = clampedSecond
    first = total - second
    if (clampSize(first, a) !== first) return start
  }
  const next = [...start]
  next[index] = round(first)
  next[index + 1] = round(second)
  return next
}

type GroupProps = Omit<React.ComponentProps<"div">, "dir"> & {
  orientation?: Orientation
  defaultLayout?: Layout
  /** After a user's change (a drag, a key, a collapse or an expand): the layout to keep. */
  onLayoutChanged?: (layout: Layout) => void
}

function ResizablePanelGroup({ orientation = "horizontal", defaultLayout, onLayoutChanged, className, children, style, ...props }: GroupProps) {
  const element = React.useRef<HTMLDivElement | null>(null)
  const panels = React.useRef(new Map<string, PanelConstraints>())
  const [order, setOrder] = React.useState<string[]>([])
  const [axis, setAxis] = React.useState(0)
  const [layout, setLayout] = React.useState<Layout | null>(null)
  // the committed layout and callback, for event handlers and effects
  const layoutRef = React.useRef<Layout | null>(null)
  // the size a collapsed panel had, for expand()
  const expanded = React.useRef(new Map<string, number>())
  const initialLayout = React.useRef(defaultLayout)
  const changed = React.useRef(onLayoutChanged)
  React.useLayoutEffect(() => {
    layoutRef.current = layout
    changed.current = onLayoutChanged
  })
  const [version, setVersion] = React.useState(0)

  const ordered = () => order.map((id) => panels.current.get(id)).filter((panel): panel is PanelConstraints => Boolean(panel))
  const sizesOf = (current: Layout) => ordered().map((panel) => current[panel.id] ?? 0)
  const toLayout = (sizes: number[]) => Object.fromEntries(ordered().map((panel, index) => [panel.id, sizes[index]!]))

  const register = React.useCallback((panel: PanelConstraints) => {
    panels.current.set(panel.id, panel)
    setVersion((v) => v + 1)
    return () => {
      panels.current.delete(panel.id)
      setVersion((v) => v + 1)
    }
  }, [])

  // the panel order (DOM order) and the group's length along its axis
  React.useLayoutEffect(() => {
    const group = element.current
    if (!group) return
    const ids = [...group.querySelectorAll<HTMLElement>(":scope > [data-panel]")].map((panel) => panel.id)
    setOrder((current) => (current.join("\n") === ids.join("\n") ? current : ids))
  }, [version])
  React.useLayoutEffect(() => {
    const group = element.current
    if (!group) return
    const measure = () => setAxis(orientation === "horizontal" ? group.offsetWidth : group.offsetHeight)
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(group)
    return () => observer.disconnect()
  }, [orientation])

  // the layout: the kept one if it fits these panels, else the panels' default sizes; re-validated
  // when the group's length changes (pixel constraints are other percentages then)
  const previousAxis = React.useRef(0)
  React.useLayoutEffect(() => {
    const list = ordered()
    if (list.length === 0 || list.length !== order.length) return
    const current = layoutRef.current
    let sizes: number[]
    if (current && list.every((panel) => panel.id in current)) {
      sizes = sizesOf(current)
      // panels that keep their pixel size take the same pixels of the new length
      if (previousAxis.current > 0 && axis > 0 && previousAxis.current !== axis) {
        sizes = sizes.map((size, index) => (list[index]!.preservePixels ? (size * previousAxis.current) / axis : size))
      }
    } else {
      const kept = initialLayout.current
      if (kept && list.every((panel) => typeof kept[panel.id] === "number") && Object.keys(kept).length === list.length) sizes = list.map((panel) => kept[panel.id]!)
      else {
        const given = list.map((panel) => (panel.defaultSize === undefined ? null : percent(panel.defaultSize, axis, 0)))
        const rest = 100 - given.reduce<number>((sum, size) => sum + (size ?? 0), 0)
        const open = given.filter((size) => size === null).length
        sizes = given.map((size) => size ?? (open ? rest / open : 0))
      }
    }
    previousAxis.current = axis
    const next = toLayout(axis > 0 ? validate(sizes, list, axis) : sizes.map(round))
    if (!current || Object.keys(next).some((id) => next[id] !== current[id]) || Object.keys(current).length !== list.length) setLayout(next)
    // oxlint-disable-next-line react/exhaustive-deps -- recomputed when the panels or the length change
  }, [order, axis, version])

  // panels hear about their size after each change (not before the group has a size: hidden or not laid out)
  const reported = React.useRef(new Map<string, string>())
  React.useEffect(() => {
    if (!layout || axis <= 0) return
    for (const panel of ordered()) {
      const asPercentage = layout[panel.id] ?? 0
      const size = { asPercentage, inPixels: round((asPercentage / 100) * axis) }
      const key = `${size.asPercentage}:${size.inPixels}`
      if (reported.current.get(panel.id) === key) continue
      reported.current.set(panel.id, key)
      panel.onResize?.(size, panel.id)
    }
  })

  const commit = (next: Layout) => {
    setLayout(next)
    changed.current?.(next)
  }

  const panelHandle = (id: string): PanelImperativeHandle => {
    const indexOf = () => order.indexOf(id)
    const boundsOf = () => bounds(panels.current.get(id)!, axis)
    const neighbour = (index: number) => (index + 1 < order.length ? index + 1 : index - 1)
    // gives `size` to panel `index`, taking the difference from (or giving it to) its neighbour
    const setSize = (index: number, size: number) => {
      const current = layoutRef.current
      if (!current || index < 0) return
      const sizes = sizesOf(current)
      const other = neighbour(index)
      if (other < 0) return
      const first = Math.min(index, other)
      const delta = index === first ? size - sizes[index]! : sizes[index]! - size
      const next = moveBoundary(sizes, first, delta, ordered(), axis)
      commit(toLayout(next))
    }
    return {
      collapse() {
        const index = indexOf()
        const current = layoutRef.current
        const b = boundsOf()
        if (!current || !b.collapsible || isCollapsedSize(current[id] ?? 0, b)) return
        expanded.current.set(id, current[id] ?? b.min)
        setSize(index, b.collapsed)
      },
      expand() {
        const index = indexOf()
        const current = layoutRef.current
        const b = boundsOf()
        if (!current || !isCollapsedSize(current[id] ?? 0, b)) return
        const panel = panels.current.get(id)!
        setSize(index, expanded.current.get(id) ?? Math.max(b.min, percent(panel.defaultSize, axis, b.min)))
      },
      isCollapsed() {
        const current = layoutRef.current
        return Boolean(current) && isCollapsedSize(current![id] ?? 0, boundsOf())
      },
      getSize() {
        const asPercentage = layoutRef.current?.[id] ?? 0
        return { asPercentage, inPixels: round((asPercentage / 100) * axis) }
      },
      resize(size: Size) {
        setSize(indexOf(), percent(size, axis, 0))
      },
    }
  }

  /** The panel index before a separator element (its position among the group's children). */
  const separatorIndex = (separator: HTMLElement): number => {
    let index = -1
    for (const child of element.current?.children ?? []) {
      if (child === separator) return index
      if ((child as HTMLElement).dataset.panel !== undefined) index += 1
    }
    return -1
  }

  const separatorProps = (separator: HTMLElement | null): SeparatorState | null => {
    if (!separator || !layout) return null
    const index = separatorIndex(separator)
    const panel = ordered()[index]
    if (!panel || index + 1 >= order.length) return null
    const b = bounds(panel, axis)
    return { controls: panel.id, now: Math.round(layout[panel.id] ?? 0), min: Math.round(b.collapsible ? b.collapsed : b.min), max: Math.round(b.max) }
  }

  const startDrag = (separator: HTMLElement, event: React.PointerEvent) => {
    const current = layoutRef.current
    const index = separatorIndex(separator)
    if (!current || index < 0 || index + 1 >= order.length || axis <= 0) return
    event.preventDefault()
    separator.focus({ preventScroll: true })
    const start = sizesOf(current)
    const origin = orientation === "horizontal" ? event.clientX : event.clientY
    const list = ordered()
    separator.setPointerCapture?.(event.pointerId)
    separator.dataset.separator = "active"
    const cursor = document.body.style.cursor
    document.body.style.cursor = orientation === "horizontal" ? "col-resize" : "row-resize"
    let last = current
    const move = (moveEvent: PointerEvent) => {
      const position = orientation === "horizontal" ? moveEvent.clientX : moveEvent.clientY
      last = toLayout(moveBoundary(start, index, ((position - origin) / axis) * 100, list, axis))
      setLayout(last)
    }
    const up = () => {
      separator.removeEventListener("pointermove", move)
      separator.removeEventListener("pointerup", up)
      separator.removeEventListener("pointercancel", up)
      separator.dataset.separator = "inactive"
      document.body.style.cursor = cursor
      changed.current?.(last)
    }
    separator.addEventListener("pointermove", move)
    separator.addEventListener("pointerup", up)
    separator.addEventListener("pointercancel", up)
  }

  const keyResize = (separator: HTMLElement, event: React.KeyboardEvent) => {
    const current = layoutRef.current
    const index = separatorIndex(separator)
    if (!current || index < 0 || index + 1 >= order.length) return
    const forward = orientation === "horizontal" ? "ArrowRight" : "ArrowDown"
    const backward = orientation === "horizontal" ? "ArrowLeft" : "ArrowUp"
    const sizes = sizesOf(current)
    const list = ordered()
    let delta: number | null = null
    if (event.key === forward) delta = 5
    else if (event.key === backward) delta = -5
    else if (event.key === "End") delta = 100
    else if (event.key === "Home") delta = -100
    else if (event.key === "Enter") {
      const b = bounds(list[index]!, axis)
      if (b.collapsible) {
        const id = list[index]!.id
        if (isCollapsedSize(sizes[index]!, b)) delta = (expanded.current.get(id) ?? b.min) - sizes[index]!
        else {
          expanded.current.set(id, sizes[index]!)
          delta = b.collapsed - sizes[index]!
        }
      }
    } else if (event.key === "F6") {
      event.preventDefault()
      const separators = [...(element.current?.querySelectorAll<HTMLElement>(":scope > [role=separator]") ?? [])]
      const at = separators.indexOf(separator)
      const next = event.shiftKey ? (at - 1 + separators.length) % separators.length : (at + 1) % separators.length
      separators[next]?.focus({ preventScroll: true })
      return
    }
    if (delta === null) return
    event.preventDefault()
    const next = moveBoundary(sizes, index, delta, list, axis)
    if (next.some((size, at) => size !== sizes[at])) commit(toLayout(next))
  }

  const resetSeparator = (separator: HTMLElement) => {
    const current = layoutRef.current
    const index = separatorIndex(separator)
    const panel = ordered()[index]
    if (!current || !panel || panel.defaultSize === undefined || index + 1 >= order.length) return
    const sizes = sizesOf(current)
    const next = moveBoundary(sizes, index, percent(panel.defaultSize, axis, sizes[index]!) - sizes[index]!, ordered(), axis)
    if (next.some((size, at) => size !== sizes[at])) commit(toLayout(next))
  }

  return (
    <GroupContext value={{ orientation, layout, register, panelHandle, separatorProps, startDrag, keyResize, resetSeparator, version: version + (layout ? 1 : 0) }}>
      <div
        ref={element}
        data-slot="resizable-panel-group"
        data-group=""
        data-orientation={orientation}
        className={cn("flex h-full w-full data-[orientation=vertical]:flex-col", className)}
        style={{ ...style, flexDirection: orientation === "horizontal" ? "row" : "column", overflow: "hidden" }}
        {...props}
      >
        {children}
      </div>
    </GroupContext>
  )
}

type PanelProps = Omit<React.ComponentProps<"div">, "id"> & {
  id: string
  defaultSize?: Size
  minSize?: Size
  maxSize?: Size
  collapsible?: boolean
  collapsedSize?: Size
  /** "preserve-pixel-size": keeps its pixels (not its share) when the group's length changes. */
  groupResizeBehavior?: "preserve-relative-size" | "preserve-pixel-size"
  onResize?: (size: PanelSize, id: string) => void
  panelRef?: React.Ref<PanelImperativeHandle | null>
}

function ResizablePanel({ id, defaultSize, minSize, maxSize, collapsible = false, collapsedSize = 0, groupResizeBehavior, onResize, panelRef, style, children, ...props }: PanelProps) {
  const group = React.useContext(GroupContext)
  if (!group) throw new Error("<ResizablePanel> must be inside <ResizablePanelGroup>")
  const resized = React.useRef(onResize)
  React.useLayoutEffect(() => {
    resized.current = onResize
  })
  const { register } = group
  React.useLayoutEffect(
    () =>
      register({
        id,
        defaultSize,
        minSize,
        maxSize,
        collapsible,
        collapsedSize,
        preservePixels: groupResizeBehavior === "preserve-pixel-size",
        onResize: (size, panelId) => resized.current?.(size, panelId),
      }),
    [register, id, defaultSize, minSize, maxSize, collapsible, collapsedSize, groupResizeBehavior],
  )
  React.useImperativeHandle(panelRef, () => group.panelHandle(id))
  // until the group has measured itself the panel takes its default share (no layout jump on the first paint)
  const size = group.layout?.[id] ?? (typeof defaultSize === "string" ? Number.parseFloat(defaultSize) : 1)
  return (
    <div
      id={id}
      data-slot="resizable-panel"
      data-panel=""
      style={{ ...style, flexBasis: 0, flexGrow: size, flexShrink: 1, overflow: "hidden" }}
      {...props}
    >
      {children}
    </div>
  )
}

function ResizableHandle({ withHandle, className, onPointerDown, onKeyDown, onDoubleClick, ...props }: React.ComponentProps<"div"> & { withHandle?: boolean }) {
  const group = React.useContext(GroupContext)
  if (!group) throw new Error("<ResizableHandle> must be inside <ResizablePanelGroup>")
  const [element, setElement] = React.useState<HTMLDivElement | null>(null)
  const state = group.separatorProps(element)
  return (
    <div
      ref={setElement}
      role="separator"
      tabIndex={0}
      aria-orientation={group.orientation === "horizontal" ? "vertical" : "horizontal"}
      aria-controls={state?.controls}
      aria-valuenow={state?.now}
      aria-valuemin={state?.min}
      aria-valuemax={state?.max}
      data-slot="resizable-handle"
      data-separator="inactive"
      className={cn(
        "relative flex w-px shrink-0 touch-none items-center justify-center bg-border ring-offset-background select-none after:absolute after:inset-y-0 after:left-1/2 after:w-1 after:-translate-x-1/2 focus-visible:ring-1 focus-visible:ring-ring focus-visible:outline-hidden aria-[orientation=horizontal]:h-px aria-[orientation=horizontal]:w-full aria-[orientation=horizontal]:after:left-0 aria-[orientation=horizontal]:after:h-1 aria-[orientation=horizontal]:after:w-full aria-[orientation=horizontal]:after:translate-x-0 aria-[orientation=horizontal]:after:-translate-y-1/2 [&[aria-orientation=horizontal]>div]:rotate-90",
        group.orientation === "horizontal" ? "cursor-col-resize" : "cursor-row-resize",
        className,
      )}
      onPointerDown={(event) => {
        onPointerDown?.(event)
        if (!event.defaultPrevented && event.button === 0) group.startDrag(event.currentTarget, event)
      }}
      onKeyDown={(event) => {
        onKeyDown?.(event)
        if (!event.defaultPrevented) group.keyResize(event.currentTarget, event)
      }}
      onDoubleClick={(event) => {
        onDoubleClick?.(event)
        if (!event.defaultPrevented) group.resetSeparator(event.currentTarget)
      }}
      {...props}
    >
      {withHandle && <div className="z-10 flex h-6 w-1 shrink-0 rounded-lg bg-border" />}
    </div>
  )
}

/** A ref for a panel's imperative handle (collapse, expand, …). */
function usePanelRef() {
  return React.useRef<PanelImperativeHandle | null>(null)
}

/**
 * The layout kept for a group (react-resizable-panels' `useDefaultLayout`: the same storage key
 * and format, so layouts saved before carry over): pass `defaultLayout` and `onLayoutChanged` to
 * the group.
 */
function useDefaultLayout({ id, panelIds, storage }: { id: string; panelIds: string[]; storage?: Pick<Storage, "getItem" | "setItem"> }) {
  const key = `react-resizable-panels:${[id, ...panelIds].join(":")}`
  const store = storage ?? (typeof localStorage === "undefined" ? undefined : localStorage)
  const defaultLayout = React.useMemo(() => {
    try {
      const saved = JSON.parse(store?.getItem(key) ?? "null") as unknown
      if (saved && typeof saved === "object" && Object.values(saved).every((value) => typeof value === "number")) return saved as Layout
    } catch {
      // a damaged entry: the default sizes apply
    }
    return undefined
  }, [store, key])
  const onLayoutChanged = React.useCallback(
    (layout: Layout) => {
      try {
        store?.setItem(key, JSON.stringify(layout))
      } catch {
        // storage full or blocked: the layout is not kept
      }
    },
    [store, key],
  )
  return { defaultLayout, onLayoutChanged }
}

export { ResizableHandle, ResizablePanel, ResizablePanelGroup, useDefaultLayout, usePanelRef }
