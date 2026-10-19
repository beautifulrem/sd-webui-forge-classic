import { afterEach, describe, expect, it, vi } from "vitest"
import { pasteTarget, registerPasteTarget, type PasteTarget } from "./paste-targets"

function field(empty: boolean, deliver = vi.fn()): PasteTarget & { node: HTMLElement; deliver: ReturnType<typeof vi.fn> } {
  const node = document.createElement("div")
  document.body.append(node)
  return { node, element: () => node, empty: () => empty, deliver }
}

function paste(files: File[], { handled = false } = {}) {
  const event = new Event("paste", { cancelable: true }) as ClipboardEvent
  Object.defineProperty(event, "clipboardData", { value: { files } })
  if (handled) event.preventDefault()
  window.dispatchEvent(event)
}

const png = () => new File(["x"], "a.png", { type: "image/png" })

describe("paste targets", () => {
  const offs: (() => void)[] = []
  afterEach(() => {
    offs.splice(0).forEach((off) => off())
    document.body.innerHTML = ""
    vi.useRealTimers()
  })

  it("picks the first empty field on screen in document order, else the last one", () => {
    const full = field(false)
    const empty = field(true)
    const later = field(true)
    expect(pasteTarget([later, full, empty])).toBe(empty)
    const hidden = field(true)
    // jsdom has no layout: what the browser answers for a display:none field
    hidden.node.checkVisibility = () => false
    expect(pasteTarget([hidden, full])).toBe(full)
    expect(pasteTarget([full, field(false)])).not.toBe(full) // all full: the last one
    expect(pasteTarget([])).toBeNull()
  })

  it("delivers an image paste nobody handled to that field, and leaves handled ones alone", () => {
    vi.useFakeTimers()
    const full = field(false)
    const empty = field(true)
    offs.push(registerPasteTarget(full), registerPasteTarget(empty))
    const file = png()
    paste([file])
    vi.runAllTimers()
    expect(empty.deliver).toHaveBeenCalledWith([file])
    expect(full.deliver).not.toHaveBeenCalled()

    paste([png()], { handled: true }) // a hovered field, the editor or PNG Info took it
    paste([new File(["t"], "a.txt", { type: "text/plain" })]) // not an image
    vi.runAllTimers()
    expect(empty.deliver).toHaveBeenCalledTimes(1)
  })
})
