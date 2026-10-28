"use client"

/**
 * Select with shadcn's API whose implementation (Base UI Select,
 * `select-impl.tsx`) loads on demand. Until a select is used it renders a
 * stand-in of its closed state: the same trigger (a `combobox` button showing
 * the selected item's label) without the listbox code. The pointer entering
 * it or focus fetches the implementation; a press (or an opening key) puts
 * the real select in the stand-in's place and opens it (as a key press on its
 * trigger does), and keyboard focus puts it there with the focus. A stand-in is not replaced under a pointer that is
 * pressing it, so a click is never lost. Forms show many selects in their
 * first view but open them later, so a page's first load does not carry the
 * select (docs/research/ui-performance.md, "Bundle").
 */
import type { Select as SelectPrimitive } from "@base-ui/react/select"
import { ChevronDownIcon } from "@/components/icons"
import * as React from "react"
import { cn } from "cn"
import { lazyModule, prefetch, standInFocus } from "@/lib/lazy-module"
import type * as Impl from "./select-impl"
import { SELECT_TRIGGER_CLASS, SELECT_VALUE_CLASS } from "./select-styles"

const implementation = lazyModule<typeof Impl>(() => import("./select-impl"))

type AnyRootProps = SelectPrimitive.Root.Props<unknown, boolean | undefined>

type RootState = {
  /** Identifies the select across the swap from the stand-in to the implementation. */
  key: string
  /** The root's props while the stand-in renders; null once the implementation does. */
  standIn: AnyRootProps | null
  /** Replaces the stand-in once the implementation is there: opened (a press or an opening key) or focused (keyboard focus). */
  activate: (how: "open" | "focus") => void
  /** The real trigger opens itself once mounted (it replaced a pressed stand-in); cleared when done. */
  takeOpen: () => boolean
}

const RootContext = React.createContext<RootState | null>(null)

function Select<Value, Multiple extends boolean | undefined = false>(props: SelectPrimitive.Root.Props<Value, Multiple>) {
  const module = implementation.useModule()
  const key = React.useId()
  const [activation, setActivation] = React.useState<"open" | "focus" | null>(null)
  const pendingOpen = React.useRef(false)
  const activate = (how: "open" | "focus") => {
    if (how === "open") pendingOpen.current = true
    setActivation((current) => current ?? how)
    prefetch(implementation)
  }
  const takeOpen = () => {
    const open = pendingOpen.current
    pendingOpen.current = false
    return open
  }
  if (!module || !activation) return <RootContext value={{ key, standIn: props as AnyRootProps, activate, takeOpen }}>{props.children}</RootContext>
  const Root = module.Select as unknown as React.ComponentType<SelectPrimitive.Root.Props<Value, Multiple>>
  return (
    <RootContext value={{ key, standIn: null, activate, takeOpen }}>
      <Root {...props} />
    </RootContext>
  )
}

function useRoot(part: string): RootState {
  const root = React.useContext(RootContext)
  if (!root) throw new Error(`<${part}> must be inside <Select>`)
  return root
}

type TriggerProps = SelectPrimitive.Trigger.Props & { size?: "sm" | "default" }

/** The label the closed select shows for its value (Base UI's `Select.Value` reads `items` the same way). */
function labelOf(props: AnyRootProps, value: unknown): React.ReactNode {
  const { items, isItemEqualToValue } = props as AnyRootProps & { isItemEqualToValue?: (a: unknown, b: unknown) => boolean }
  const equal = isItemEqualToValue ?? Object.is
  if (Array.isArray(items)) {
    for (const item of items as { value: unknown; label: React.ReactNode }[]) {
      if (item && typeof item === "object" && "value" in item && equal(item.value, value)) return item.label
    }
  } else if (items && typeof items === "object" && typeof value === "string" && value in items) {
    return (items as Record<string, React.ReactNode>)[value]
  }
  return value === null || value === undefined ? null : String(value)
}

function selectedValue(props: AnyRootProps): unknown {
  return props.value !== undefined ? props.value : (props.defaultValue ?? null)
}

function hasValue(value: unknown): boolean {
  return Array.isArray(value) ? value.length > 0 : value !== null && value !== undefined
}

function StandInTrigger({ root, className, size = "default", children, ...props }: TriggerProps & { root: RootState }) {
  const standIn = root.standIn as AnyRootProps
  const disabled = Boolean(standIn.disabled || props.disabled)
  const placeholder = !hasValue(selectedValue(standIn))
  const state = { open: false, disabled, readOnly: Boolean(standIn.readOnly), placeholder } as never
  const { render: _render, nativeButton: _nativeButton, style, ref, ...rest } = props
  // a press focuses the button before its click: that focus must not replace the button under the pointer
  const pressing = React.useRef(false)
  return (
    <button
      {...(rest as React.ComponentProps<"button">)}
      ref={ref as React.Ref<HTMLButtonElement>}
      type="button"
      role="combobox"
      aria-expanded={false}
      aria-haspopup="listbox"
      data-slot="select-trigger"
      data-size={size}
      data-placeholder={placeholder ? "" : undefined}
      data-disabled={disabled ? "" : undefined}
      disabled={disabled}
      style={typeof style === "function" ? style(state) : style}
      className={cn(SELECT_TRIGGER_CLASS, typeof className === "function" ? className(state) : className)}
      onPointerEnter={(event) => {
        prefetch(implementation)
        rest.onPointerEnter?.(event as never)
      }}
      onPointerDown={(event) => {
        pressing.current = true
        rest.onPointerDown?.(event as never)
      }}
      onFocus={(event) => {
        standInFocus.focused(root.key)
        if (pressing.current) prefetch(implementation)
        else root.activate("focus")
        rest.onFocus?.(event as never)
      }}
      onBlur={(event) => {
        pressing.current = false
        standInFocus.blurred(root.key)
        rest.onBlur?.(event as never)
      }}
      onClick={(event) => {
        pressing.current = false
        rest.onClick?.(event as never)
        if (!event.defaultPrevented) root.activate("open")
      }}
      onKeyDown={(event) => {
        rest.onKeyDown?.(event as never)
        if (!event.defaultPrevented && (event.key === "ArrowDown" || event.key === "ArrowUp")) {
          event.preventDefault()
          root.activate("open")
        }
      }}
    >
      {children}
      <ChevronDownIcon aria-hidden className="pointer-events-none size-4 text-muted-foreground" />
    </button>
  )
}

function LoadedTrigger({ root, ref, ...props }: TriggerProps & { root: RootState }) {
  const Trigger = implementation.get()!.SelectTrigger
  const element = React.useRef<HTMLButtonElement | null>(null)
  React.useLayoutEffect(() => {
    if (standInFocus.take(root.key)) element.current?.focus()
  }, [root.key])
  // the stand-in was pressed: the real trigger opens the select the way a key press on it does, so
  // the select's own focus handling (back to this trigger on close) and open events apply
  React.useEffect(() => {
    if (!root.takeOpen()) return
    element.current?.focus()
    element.current?.click()
  }, [root])
  const setRef = (node: HTMLButtonElement | null) => {
    element.current = node
    if (typeof ref === "function") ref(node)
    else if (ref) (ref as React.RefObject<HTMLButtonElement | null>).current = node
  }
  return <Trigger {...props} ref={setRef} />
}

function SelectTrigger(props: TriggerProps) {
  const root = useRoot("SelectTrigger")
  return root.standIn ? <StandInTrigger root={root} {...props} /> : <LoadedTrigger root={root} {...props} />
}

function SelectValue({ className, children, placeholder, ...props }: SelectPrimitive.Value.Props) {
  const root = useRoot("SelectValue")
  if (!root.standIn) {
    const Value = implementation.get()!.SelectValue
    return (
      <Value className={className} placeholder={placeholder} {...props}>
        {children}
      </Value>
    )
  }
  const value = selectedValue(root.standIn)
  const empty = !hasValue(value)
  let content: React.ReactNode
  if (typeof children === "function") content = (children as (value: unknown) => React.ReactNode)(value)
  else if (children !== undefined) content = children
  else if (empty) content = placeholder
  else if (Array.isArray(value)) content = value.map((v) => labelOf(root.standIn!, v)).join(", ")
  else content = labelOf(root.standIn, value)
  const state = { value, placeholder: empty } as never
  return (
    <span
      data-slot="select-value"
      data-placeholder={empty ? "" : undefined}
      className={cn(SELECT_VALUE_CLASS, typeof className === "function" ? className(state) : className)}
      style={typeof props.style === "function" ? props.style(state) : props.style}
      id={props.id}
    >
      {content}
    </span>
  )
}

/** A part that only renders inside the open select: nothing while the stand-in renders. */
function deferred<K extends keyof typeof Impl>(name: K): (typeof Impl)[K] {
  function Part(props: Record<string, unknown>) {
    const root = React.useContext(RootContext)
    const module = implementation.useModule()
    if (!module || root?.standIn) return null
    const Component = module[name] as unknown as React.ComponentType<Record<string, unknown>>
    return <Component {...props} />
  }
  Part.displayName = name
  return Part as unknown as (typeof Impl)[K]
}

const SelectContent = deferred("SelectContent")
const SelectGroup = deferred("SelectGroup")
const SelectItem = deferred("SelectItem")
const SelectLabel = deferred("SelectLabel")
const SelectSeparator = deferred("SelectSeparator")
const SelectScrollUpButton = deferred("SelectScrollUpButton")
const SelectScrollDownButton = deferred("SelectScrollDownButton")

export {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectScrollDownButton,
  SelectScrollUpButton,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
}
