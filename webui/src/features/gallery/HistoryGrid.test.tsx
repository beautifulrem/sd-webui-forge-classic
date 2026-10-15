import { fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { HistoryGrid } from "./HistoryGrid"
import { useSession } from "./session-store"

const outputs = (n: number) => Array.from({ length: n }, (_, i) => ({ id: `f${i}`, kind: "image" as const }))

/** jsdom has no layout: give the scroller a size so the virtualizer renders rows. */
function layout(width: number, height: number) {
  vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockReturnValue(width)
  vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockReturnValue(width)
  vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockReturnValue(height)
}

describe("history grid", () => {
  afterEach(() => vi.restoreAllMocks())
  beforeEach(() => {
    useSession.getState().clear()
    // Newest first: job b's results come before job a's.
    useSession.getState().addJob({ jobId: "a", tab: "txt2img", outputs: outputs(2) })
    useSession.getState().addJob({ jobId: "b", tab: "img2img", outputs: outputs(2) })
  })

  it("moves the selection with the arrow keys, opens with Enter and removes with Delete", () => {
    layout(220, 400)
    const onOpen = vi.fn()
    render(<HistoryGrid onOpen={onOpen} />)
    const list = screen.getByRole("listbox", { name: "History" })
    expect(useSession.getState().selected).toBe("b:0")
    expect(list).toHaveAttribute("aria-activedescendant", screen.getByRole("option", { name: /Result 1/ }).id)

    fireEvent.keyDown(list, { key: "ArrowRight" })
    expect(useSession.getState().selected).toBe("b:1")
    // Two columns: down skips a row.
    fireEvent.keyDown(list, { key: "ArrowDown" })
    expect(useSession.getState().selected).toBe("a:1")
    fireEvent.keyDown(list, { key: "Home" })
    expect(useSession.getState().selected).toBe("b:0")
    expect(useSession.getState().follow).toBe(false)

    fireEvent.keyDown(list, { key: "Enter" })
    expect(onOpen).toHaveBeenCalledWith("b:0")

    fireEvent.keyDown(list, { key: "Delete" })
    expect(useSession.getState().items.map((item) => item.key)).toEqual(["b:1", "a:0", "a:1"])
    expect(useSession.getState().selected).toBe("b:1")
  })

  it("does not point aria-activedescendant at an option that is not rendered", () => {
    layout(0, 0)
    render(<HistoryGrid onOpen={() => {}} />)
    expect(screen.queryAllByRole("option")).toHaveLength(0)
    expect(screen.getByRole("listbox")).not.toHaveAttribute("aria-activedescendant")
  })

  it("ignores keys with modifiers (shortcuts of the page)", () => {
    render(<HistoryGrid onOpen={() => {}} />)
    fireEvent.keyDown(screen.getByRole("listbox"), { key: "ArrowRight", ctrlKey: true })
    expect(useSession.getState().selected).toBe("b:0")
  })
})
