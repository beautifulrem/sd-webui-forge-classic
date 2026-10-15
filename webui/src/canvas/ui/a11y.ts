/** Keyboard accessibility helpers of the editor chrome. */
import type { KeyboardEvent as ReactKeyboardEvent } from "react"

/** Shown key names → `aria-keyshortcuts` key names (KeyboardEvent.key values, modifiers spelled out). */
const ARIA_KEYS: Record<string, string> = { "⌘": "Meta", Ctrl: "Control", "⇧": "Shift", "−": "Minus", "-": "Minus", "+": "Plus", " ": "Space" }

/** The `aria-keyshortcuts` value of a key combination as shown in a tooltip (e.g. `["⌘", "Z"]` → `"Meta+Z"`). */
export function ariaKeyShortcuts(keys: readonly string[] | undefined): string | undefined {
  if (!keys || keys.length === 0) return undefined
  return keys.map((key) => ARIA_KEYS[key] ?? key).join("+")
}

/**
 * Arrow-key navigation between the buttons of a toolbar (WAI-ARIA toolbar
 * pattern): Up/Down (vertical) or Left/Right (horizontal) move the focus,
 * Home/End go to the ends. Sliders and text fields keep their arrow keys.
 */
export function toolbarKeys(orientation: "horizontal" | "vertical") {
  const prev = orientation === "vertical" ? "ArrowUp" : "ArrowLeft"
  const next = orientation === "vertical" ? "ArrowDown" : "ArrowRight"
  return (event: ReactKeyboardEvent<HTMLElement>) => {
    if (![prev, next, "Home", "End"].includes(event.key) || event.altKey || event.ctrlKey || event.metaKey) return
    const target = event.target as HTMLElement
    if (target.closest("[role=slider], input, textarea, [contenteditable=true]")) return
    const items = Array.from(event.currentTarget.querySelectorAll<HTMLElement>("button:not([disabled])")).filter((item) => item.checkVisibility?.() ?? true)
    const index = items.indexOf(target.closest("button") as HTMLElement)
    if (index < 0 || items.length === 0) return
    event.preventDefault()
    const to = event.key === "Home" ? 0 : event.key === "End" ? items.length - 1 : (index + (event.key === next ? 1 : -1) + items.length) % items.length
    items[to]?.focus()
  }
}
