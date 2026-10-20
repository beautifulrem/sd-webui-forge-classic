"use client"

/**
 * Dropdown menu with shadcn's API whose implementation (Base UI Menu,
 * `dropdown-menu-impl.tsx`) loads on demand: the trigger is rendered here, a
 * button announced as a menu button, and the menu code is fetched when the
 * pointer enters it, it gets focus or the menu opens. Pages show menu buttons
 * in their first view (result actions, "More actions") but open them only
 * later, so their first load does not carry the menu
 * (docs/research/ui-performance.md, "Bundle").
 *
 * The popup is anchored to the trigger and returns focus to it on close;
 * menu parts (items, separators, ...) render inside the content, after the
 * implementation has loaded.
 */
import { mergeProps } from "@base-ui/react/merge-props"
import * as React from "react"
import { cn } from "cn"
import type * as Impl from "./dropdown-menu-impl"

type Module = typeof Impl

let loaded: Module | null = null
let loading: Promise<Module> | null = null
const listeners = new Set<() => void>()

/** Fetches the menu implementation (once); resolves with it. */
function loadDropdownMenu(): Promise<Module> {
  loading ??= import("./dropdown-menu-impl").then((module) => {
    loaded = module
    for (const listener of listeners) listener()
    return module
  })
  return loading
}

function subscribe(listener: () => void) {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

const getLoaded = () => loaded

function useImplementation(): Module | null {
  return React.useSyncExternalStore(subscribe, getLoaded, getLoaded)
}

type RootProps = React.ComponentProps<Module["DropdownMenu"]>
type ChangeDetails = Parameters<NonNullable<RootProps["onOpenChange"]>>[1]

type MenuState = {
  open: boolean
  setOpen: (open: boolean, details?: ChangeDetails) => void
  /** The trigger's press: opens or closes (not again right after the same press closed the menu). */
  toggle: () => void
  root: Omit<RootProps, "open" | "defaultOpen" | "onOpenChange" | "children">
  /** The trigger element: the popup's anchor and where focus returns. */
  anchor: HTMLElement | null
  setAnchor: (element: HTMLElement | null) => void
  triggerId: string
  popupId: string
}

const MenuContext = React.createContext<MenuState | null>(null)

function useMenu(part: string): MenuState {
  const menu = React.useContext(MenuContext)
  if (!menu) throw new Error(`<${part}> must be inside <DropdownMenu>`)
  return menu
}

/** Whether a pointer event happened over the element (a modal menu's backdrop covers the trigger). */
function pressedOn(element: HTMLElement | null, event: Event | undefined): boolean {
  if (!element || !event || !("clientX" in event)) return false
  const { clientX: x, clientY: y } = event as MouseEvent
  const rect = element.getBoundingClientRect()
  return x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom
}

function DropdownMenu({ open: openProp, defaultOpen = false, onOpenChange, children, ...root }: RootProps) {
  const [ownOpen, setOwnOpen] = React.useState(defaultOpen)
  const open = openProp ?? ownOpen
  const [anchor, setAnchor] = React.useState<HTMLElement | null>(null)
  // when a press on the trigger (an outside press for the open popup) last closed the menu: the
  // click that follows must not open it again
  const closedAt = React.useRef(0)
  const triggerId = React.useId()
  const popupId = React.useId()
  React.useEffect(() => {
    if (open) void loadDropdownMenu()
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
  return <MenuContext value={{ open, setOpen, toggle, root, anchor, setAnchor, triggerId, popupId }}>{children as React.ReactNode}</MenuContext>
}

type TriggerProps = React.ComponentProps<Module["DropdownMenuTrigger"]>

/** One callback ref that sets every given ref (the menu's anchor, the consumer's, the render element's). */
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

function DropdownMenuTrigger({
  render,
  children,
  ref,
  className,
  style,
  // Base UI trigger options without meaning here (the menu opens on press)
  nativeButton: _nativeButton,
  openOnHover: _openOnHover,
  delay: _delay,
  closeDelay: _closeDelay,
  handle: _handle,
  payload: _payload,
  ...props
}: TriggerProps) {
  const menu = useMenu("DropdownMenuTrigger")
  const state = { open: menu.open, disabled: Boolean(props.disabled) } as never
  const elementRef = React.isValidElement(render) ? (render.props as { ref?: React.Ref<HTMLElement> }).ref : undefined
  const setRef = useMergedRef<HTMLElement>(menu.setAnchor, ref as React.Ref<HTMLElement> | undefined, elementRef)
  const own = {
    id: menu.triggerId,
    type: "button" as const,
    "data-slot": "dropdown-menu-trigger",
    "aria-haspopup": "menu" as const,
    "aria-expanded": menu.open,
    "aria-controls": menu.open ? menu.popupId : undefined,
    "data-popup-open": menu.open ? "" : undefined,
    className: typeof className === "function" ? className(state) : className,
    style: typeof style === "function" ? style(state) : style,
    // intent: fetch the menu before the press
    onPointerEnter: () => void loadDropdownMenu(),
    onFocus: () => void loadDropdownMenu(),
    onClick: (event: React.MouseEvent) => {
      if (!event.defaultPrevented && !props.disabled) menu.toggle()
    },
    onKeyDown: (event: React.KeyboardEvent) => {
      if (menu.open || props.disabled || (event.key !== "ArrowDown" && event.key !== "ArrowUp")) return
      event.preventDefault()
      menu.setOpen(true)
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

type ContentProps = React.ComponentProps<Module["DropdownMenuContent"]>

function DropdownMenuContent(props: ContentProps) {
  const menu = useMenu("DropdownMenuContent")
  const implementation = useImplementation()
  if (!implementation) return null
  const { DropdownMenu: Root, DropdownMenuContent: Content } = implementation
  return (
    <Root {...menu.root} open={menu.open} onOpenChange={menu.setOpen}>
      <Content id={menu.popupId} anchor={menu.anchor} finalFocus={() => menu.anchor ?? true} {...props} />
    </Root>
  )
}

/** A menu part that renders the implementation's (it only renders inside an open menu). */
function deferred<K extends keyof Module>(name: K): Module[K] {
  function Part(props: Record<string, unknown>) {
    const implementation = useImplementation()
    if (!implementation) return null
    const Component = implementation[name] as unknown as React.ComponentType<Record<string, unknown>>
    return <Component {...props} />
  }
  Part.displayName = name
  return Part as unknown as Module[K]
}

const DropdownMenuPortal = deferred("DropdownMenuPortal")
const DropdownMenuGroup = deferred("DropdownMenuGroup")
const DropdownMenuLabel = deferred("DropdownMenuLabel")
const DropdownMenuItem = deferred("DropdownMenuItem")
const DropdownMenuCheckboxItem = deferred("DropdownMenuCheckboxItem")
const DropdownMenuRadioGroup = deferred("DropdownMenuRadioGroup")
const DropdownMenuRadioItem = deferred("DropdownMenuRadioItem")
const DropdownMenuSeparator = deferred("DropdownMenuSeparator")
const DropdownMenuSub = deferred("DropdownMenuSub")
const DropdownMenuSubTrigger = deferred("DropdownMenuSubTrigger")
const DropdownMenuSubContent = deferred("DropdownMenuSubContent")

function DropdownMenuShortcut({ className, ...props }: React.ComponentProps<"span">) {
  return (
    <span
      data-slot="dropdown-menu-shortcut"
      className={cn("ml-auto text-xs tracking-widest text-muted-foreground group-focus/dropdown-menu-item:text-accent-foreground", className)}
      {...props}
    />
  )
}

export {
  DropdownMenu,
  DropdownMenuPortal,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuLabel,
  DropdownMenuItem,
  DropdownMenuCheckboxItem,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
  DropdownMenuSub,
  DropdownMenuSubTrigger,
  DropdownMenuSubContent,
  loadDropdownMenu,
}
