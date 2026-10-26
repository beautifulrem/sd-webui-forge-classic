"use client"

/**
 * Popover with shadcn's API whose implementation (Base UI Popover,
 * `popover-impl.tsx`) loads on demand: the trigger is rendered here, a
 * button announced as opening a dialog, and the popover code is fetched when
 * the pointer enters it, it gets focus or the popover opens. Pages show
 * popover buttons in their first view (aspect ratio, "Generate several
 * times", model settings) but open them only later, so their first load does
 * not carry the popover (docs/research/ui-performance.md, "Bundle"); the
 * dropdown menu does the same (`dropdown-menu.tsx`).
 *
 * The popup is anchored to the trigger and returns focus to it on close;
 * the parts inside the popup (title, description) render with it, after the
 * implementation has loaded.
 */
import { mergeProps } from "@base-ui/react/merge-props"
import * as React from "react"
import { cn } from "cn"
import { lazyModule, prefetch } from "@/lib/lazy-module"
import type * as Impl from "./popover-impl"

type Module = typeof Impl

const implementation = lazyModule<Module>(() => import("./popover-impl"))

type RootProps = React.ComponentProps<Module["Popover"]>
type ChangeDetails = Parameters<NonNullable<RootProps["onOpenChange"]>>[1]

type PopoverState = {
  open: boolean
  setOpen: (open: boolean, details?: ChangeDetails) => void
  /** The trigger's press: opens or closes (not again right after the same press closed the popover). */
  toggle: () => void
  root: Omit<RootProps, "open" | "defaultOpen" | "onOpenChange" | "children">
  /** The trigger element: the popup's anchor and where focus returns. */
  anchor: HTMLElement | null
  setAnchor: (element: HTMLElement | null) => void
  popupId: string
}

const PopoverContext = React.createContext<PopoverState | null>(null)

function usePopover(part: string): PopoverState {
  const popover = React.useContext(PopoverContext)
  if (!popover) throw new Error(`<${part}> must be inside <Popover>`)
  return popover
}

/** Whether a pointer event happened over the element (the press on the trigger that closed the popover). */
function pressedOn(element: HTMLElement | null, event: Event | undefined): boolean {
  if (!element || !event || !("clientX" in event)) return false
  const { clientX: x, clientY: y } = event as MouseEvent
  const rect = element.getBoundingClientRect()
  return x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom
}

function Popover({ open: openProp, defaultOpen = false, onOpenChange, children, ...root }: RootProps) {
  const [ownOpen, setOwnOpen] = React.useState(defaultOpen)
  const open = openProp ?? ownOpen
  const [anchor, setAnchor] = React.useState<HTMLElement | null>(null)
  // when a press on the trigger (an outside press for the open popup) last closed the popover: the
  // click that follows must not open it again
  const closedAt = React.useRef(0)
  const popupId = React.useId()
  React.useEffect(() => {
    if (open) prefetch(implementation)
  }, [open])
  const setOpen = (next: boolean, details?: ChangeDetails) => {
    if (!next && details?.reason === "outside-press" && pressedOn(anchor, details.event)) closedAt.current = performance.now()
    if (openProp === undefined) setOwnOpen(next)
    onOpenChange?.(next, details as ChangeDetails)
  }
  const toggle = () => {
    if (!open && performance.now() - closedAt.current < 400) return
    setOpen(!open)
  }
  return <PopoverContext value={{ open, setOpen, toggle, root, anchor, setAnchor, popupId }}>{children as React.ReactNode}</PopoverContext>
}

type TriggerProps = React.ComponentProps<Module["PopoverTrigger"]>

/** One callback ref that sets every given ref (the popover's anchor, the consumer's, the render element's). */
function useMergedRef<T>(...refs: (React.Ref<T> | undefined)[]): React.RefCallback<T> {
  return React.useCallback(
    (node: T | null) => {
      for (const ref of refs) {
        if (typeof ref === "function") ref(node)
        else if (ref) ref.current = node
      }
    },
    // oxlint-disable-next-line react/exhaustive-deps -- the refs themselves are the dependencies
    refs,
  )
}

function PopoverTrigger({
  render,
  children,
  ref,
  className,
  style,
  // Base UI trigger options without meaning here (the popover opens on press)
  nativeButton: _nativeButton,
  openOnHover: _openOnHover,
  delay: _delay,
  closeDelay: _closeDelay,
  handle: _handle,
  payload: _payload,
  ...props
}: TriggerProps) {
  const popover = usePopover("PopoverTrigger")
  const state = { open: popover.open, disabled: Boolean(props.disabled) } as never
  const elementRef = React.isValidElement(render) ? (render.props as { ref?: React.Ref<HTMLElement> }).ref : undefined
  const setRef = useMergedRef<HTMLElement>(popover.setAnchor, ref as React.Ref<HTMLElement> | undefined, elementRef)
  const own = {
    type: "button" as const,
    "data-slot": "popover-trigger",
    "aria-haspopup": "dialog" as const,
    "aria-expanded": popover.open,
    "aria-controls": popover.open ? popover.popupId : undefined,
    "data-popup-open": popover.open ? "" : undefined,
    className: typeof className === "function" ? className(state) : className,
    style: typeof style === "function" ? style(state) : style,
    // intent: fetch the popover before the press
    onPointerEnter: () => prefetch(implementation),
    onFocus: () => prefetch(implementation),
    onClick: (event: React.MouseEvent) => {
      if (!event.defaultPrevented && !props.disabled) popover.toggle()
    },
    children,
  }
  const merged = { ...mergeProps<"button">(own, props as React.ComponentProps<"button">), ref: setRef }
  if (typeof render === "function") return render(merged as never, state)
  if (React.isValidElement(render)) {
    const element = render as React.ReactElement<Record<string, unknown>>
    return React.cloneElement(element, { ...mergeProps<"button">(merged, element.props as React.ComponentProps<"button">), ref: setRef })
  }
  return <button {...merged} />
}

type ContentProps = React.ComponentProps<Module["PopoverContent"]>

function PopoverContent(props: ContentProps) {
  const popover = usePopover("PopoverContent")
  const module = implementation.useModule()
  if (!module) return null
  const { Popover: Root, PopoverContent: Content } = module
  return (
    <Root {...popover.root} open={popover.open} onOpenChange={popover.setOpen}>
      <Content id={popover.popupId} anchor={popover.anchor} finalFocus={() => popover.anchor ?? true} {...props} />
    </Root>
  )
}

/** A popup part that renders the implementation's (it only renders inside an open popover). */
function deferred<K extends keyof Module>(name: K): Module[K] {
  function Part(props: Record<string, unknown>) {
    const module = implementation.useModule()
    if (!module) return null
    const Component = module[name] as unknown as React.ComponentType<Record<string, unknown>>
    return <Component {...props} />
  }
  Part.displayName = name
  return Part as unknown as Module[K]
}

const PopoverTitle = deferred("PopoverTitle")
const PopoverDescription = deferred("PopoverDescription")

function PopoverHeader({ className, ...props }: React.ComponentProps<"div">) {
  return <div data-slot="popover-header" className={cn("flex flex-col gap-0.5 text-sm", className)} {...props} />
}

export { Popover, PopoverContent, PopoverDescription, PopoverHeader, PopoverTitle, PopoverTrigger }
