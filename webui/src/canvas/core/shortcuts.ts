/**
 * Editor keyboard shortcuts (pure): key events → commands, and the list the
 * help popover shows. Active while the pointer is over the editor or focus is
 * inside it, never while typing in a text field.
 */

export type Command =
  | { type: "tool"; tool: "brush" | "eraser" | "fill" | "rect" | "lasso" | "eyedropper" | "hand" }
  | { type: "undo" }
  | { type: "redo" }
  | { type: "size"; direction: 1 | -1 }
  | { type: "clear" }
  | { type: "invert" }
  | { type: "fit" }
  | { type: "actual" }
  | { type: "zoom"; direction: 1 | -1 }
  | { type: "maximize" }
  | { type: "help" }

export type KeyLike = Pick<KeyboardEvent, "key" | "code" | "ctrlKey" | "metaKey" | "altKey" | "shiftKey">

const TOOL_KEYS: Record<string, Extract<Command, { type: "tool" }>["tool"]> = {
  KeyB: "brush",
  KeyE: "eraser",
  KeyG: "fill",
  KeyM: "rect",
  KeyL: "lasso",
  KeyI: "eyedropper",
  KeyH: "hand",
}

/** The command of a key press, or null. `mac` picks Cmd over Ctrl as the modifier. */
export function commandFor(event: KeyLike, mac = false): Command | null {
  const mod = mac ? event.metaKey : event.ctrlKey
  const other = mac ? event.ctrlKey : event.metaKey
  if (other || event.altKey) return null
  if (mod) {
    switch (event.code) {
      case "KeyZ":
        return event.shiftKey ? { type: "redo" } : { type: "undo" }
      case "KeyY":
        return { type: "redo" }
      case "KeyX":
        return { type: "clear" }
      case "KeyI":
        return { type: "invert" }
      default:
        return null
    }
  }
  if (event.code in TOOL_KEYS && !event.shiftKey) return { type: "tool", tool: TOOL_KEYS[event.code]! }
  switch (event.key) {
    case "[":
      return { type: "size", direction: -1 }
    case "]":
      return { type: "size", direction: 1 }
    case "+":
    case "=":
      return { type: "zoom", direction: 1 }
    case "-":
    case "_":
      return { type: "zoom", direction: -1 }
    case "?":
      return { type: "help" }
    case "Delete":
    case "Backspace":
      return { type: "clear" }
  }
  if (event.shiftKey) return null
  switch (event.code) {
    case "KeyR":
    case "Digit0":
      return { type: "fit" }
    case "Digit1":
      return { type: "actual" }
    case "KeyF":
      return { type: "maximize" }
  }
  return null
}

export type ShortcutHelp = { keys: string[]; label: string; group: "Tools" | "Brush" | "Edit" | "View" }

/** What the help popover lists (labels are English chrome keys, translated at render). */
export function shortcutList(mac: boolean): ShortcutHelp[] {
  const mod = mac ? "⌘" : "Ctrl"
  return [
    { keys: ["B"], label: "Brush", group: "Tools" },
    { keys: ["E"], label: "Eraser", group: "Tools" },
    { keys: ["G"], label: "Fill", group: "Tools" },
    { keys: ["M"], label: "Rectangle", group: "Tools" },
    { keys: ["L"], label: "Lasso", group: "Tools" },
    { keys: ["I"], label: "Eyedropper", group: "Tools" },
    { keys: ["H"], label: "Hand", group: "Tools" },
    { keys: ["Shift"], label: "Hold to erase or subtract", group: "Brush" },
    { keys: ["[", "]"], label: "Brush size", group: "Brush" },
    { keys: ["W", "Wheel"], label: "Brush size", group: "Brush" },
    { keys: ["A", "Wheel"], label: "Brush opacity", group: "Brush" },
    { keys: ["S", "Wheel"], label: "Brush softness", group: "Brush" },
    { keys: [mod, "Z"], label: "Undo", group: "Edit" },
    { keys: [mod, "Y"], label: "Redo", group: "Edit" },
    { keys: [mod, "X"], label: "Clear layer", group: "Edit" },
    { keys: [mod, "I"], label: "Invert mask", group: "Edit" },
    { keys: ["Wheel"], label: "Zoom at the cursor", group: "View" },
    { keys: ["Space", "Drag"], label: "Pan", group: "View" },
    { keys: ["Right drag"], label: "Pan", group: "View" },
    { keys: ["R"], label: "Fit and center", group: "View" },
    { keys: ["1"], label: "Actual size", group: "View" },
    { keys: ["+", "−"], label: "Zoom in or out", group: "View" },
    { keys: ["F"], label: "Maximize", group: "View" },
  ]
}

/** Whether a key event comes from a text field (shortcuts stay out of typing). */
export function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  if (target.isContentEditable) return true
  if (target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) return true
  if (target instanceof HTMLInputElement) return !["button", "checkbox", "radio", "range", "color", "file", "submit", "reset"].includes(target.type)
  return Boolean(target.closest("[contenteditable=true], .cm-editor"))
}

/** Whether the pointer is over `element` (`:hover`; false where the selector is not supported). */
export function isHovered(element: Element): boolean {
  try {
    return element.matches(":hover")
  } catch {
    return false
  }
}

export const isMac = () => typeof navigator !== "undefined" && /Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent)
