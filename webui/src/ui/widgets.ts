/**
 * Custom widget registry. Every `src/widgets/<name>/index.tsx` is a widget
 * named `<name>` (its default export), discovered at build time and loaded
 * lazily on first use; there is no central list to edit.
 */
import { type ComponentType, lazy, type LazyExoticComponent } from "react"
import type { CustomNode, Json } from "./schema"

export type WidgetProps<V = Json, P extends Record<string, unknown> = Record<string, Json>> = {
  /** Current value (JSON). */
  value: V
  /** Commits a new value (triggers `on_change`). */
  onChange: (value: V) => void
  /** `props` from the schema node. */
  props: P
  node: CustomNode
  disabled: boolean
  /** DOM id the field label points at; put it on the primary focusable element. */
  id: string
  /** Translated label ("" when the node has none). */
  label: string
}

// oxlint-disable-next-line typescript/no-explicit-any -- widgets choose their own value/props types
type AnyWidget = ComponentType<WidgetProps<any, any>>
type WidgetModule = { default: AnyWidget }

const modules = import.meta.glob<WidgetModule>("../widgets/*/index.tsx")

function nameOf(path: string): string {
  return path.split("/").at(-2) ?? path
}

const registry = new Map<string, LazyExoticComponent<AnyWidget>>(
  Object.entries(modules).map(([path, load]) => [nameOf(path), lazy(load)]),
)

export function getWidget(name: string): LazyExoticComponent<AnyWidget> | undefined {
  return registry.get(name)
}

export function widgetNames(): string[] {
  return [...registry.keys()].sort()
}
