import { afterEach, describe, expect, it, vi } from "vitest"
import { onShortcutsKey, useShortcutsHelp } from "./shortcuts-help"

function press(target: EventTarget, init: KeyboardEventInit = { key: "?" }) {
  const event = new KeyboardEvent("keydown", { bubbles: true, cancelable: true, ...init })
  target.dispatchEvent(event)
  onShortcutsKey(Object.defineProperty(event, "target", { value: target }))
  return event
}

describe("the ? shortcut", () => {
  afterEach(() => {
    useShortcutsHelp.setState({ open: false })
    document.body.innerHTML = ""
    vi.useRealTimers()
  })

  it("opens the dialog outside text fields and dialogs, unless a handler used the key", () => {
    vi.useFakeTimers()
    press(document.body)
    vi.runAllTimers()
    expect(useShortcutsHelp.getState().open).toBe(true)

    useShortcutsHelp.setState({ open: false })
    const input = document.body.appendChild(document.createElement("input"))
    press(input)
    vi.runAllTimers()
    expect(useShortcutsHelp.getState().open).toBe(false)

    press(document.body).preventDefault() // the image editor's own "?" list
    vi.runAllTimers()
    expect(useShortcutsHelp.getState().open).toBe(false)

    document.body.appendChild(document.createElement("div")).setAttribute("role", "dialog")
    press(document.body)
    vi.runAllTimers()
    expect(useShortcutsHelp.getState().open).toBe(false)
  })
})
