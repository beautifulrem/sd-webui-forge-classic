import { afterEach, describe, expect, it } from "vitest"
import { escapeIsFree } from "./shortcuts"

/** Dispatches Escape on `target` and reports what `escapeIsFree` said at the window. */
function escapeFrom(target: Element, { prevent = false } = {}): boolean {
  let free: boolean | null = null
  const listen = (event: KeyboardEvent) => {
    free = escapeIsFree(event)
  }
  window.addEventListener("keydown", listen)
  if (prevent) target.addEventListener("keydown", (event) => event.preventDefault(), { once: true })
  target.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true }))
  window.removeEventListener("keydown", listen)
  return free!
}

/** The DOM of a prompt editor: wrapper, CodeMirror root, editable content. */
function promptEditor(parent: Element = document.body) {
  parent.insertAdjacentHTML(
    "beforeend",
    '<div data-slot="prompt-editor"><input data-testid="decorator-field"><div class="cm-editor"><div class="cm-scroller"><div class="cm-content" contenteditable="true"></div></div></div></div>',
  )
  const root = parent.lastElementChild!
  return { content: root.querySelector(".cm-content")!, decoratorField: root.querySelector("input")! }
}

describe("escapeIsFree", () => {
  afterEach(() => {
    document.body.innerHTML = ""
  })

  it("interrupts from the page and from a prompt editor (the placeholder's 'Esc to Interrupt')", () => {
    expect(escapeFrom(document.body)).toBe(true)
    expect(escapeFrom(promptEditor().content)).toBe(true)
  })

  it("leaves Escape to the prompt editor when it used it (completion list, selection)", () => {
    expect(escapeFrom(promptEditor().content, { prevent: true })).toBe(false)
  })

  it("leaves Escape to ordinary fields, a decorator's fields, dialogs and open popups", () => {
    const input = document.createElement("input")
    document.body.append(input)
    expect(escapeFrom(input)).toBe(false)
    expect(escapeFrom(promptEditor().decoratorField)).toBe(false)
    const dialog = document.createElement("div")
    dialog.setAttribute("role", "dialog")
    document.body.append(dialog)
    expect(escapeFrom(promptEditor(dialog).content)).toBe(false)
    dialog.remove()
    const popup = document.createElement("div")
    popup.setAttribute("role", "listbox")
    popup.setAttribute("data-open", "")
    document.body.append(popup)
    expect(escapeFrom(promptEditor().content)).toBe(false)
  })
})
