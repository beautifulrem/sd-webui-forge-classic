import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { ariaKeyShortcuts, toolbarKeys } from "./a11y"

describe("aria-keyshortcuts", () => {
  it("spells the shown keys the way the attribute wants them", () => {
    expect(ariaKeyShortcuts(["⌘", "Z"])).toBe("Meta+Z")
    expect(ariaKeyShortcuts(["Ctrl", "Y"])).toBe("Control+Y")
    expect(ariaKeyShortcuts(["−"])).toBe("Minus")
    expect(ariaKeyShortcuts(["+"])).toBe("Plus")
    expect(ariaKeyShortcuts(["F"])).toBe("F")
    expect(ariaKeyShortcuts([])).toBeUndefined()
  })
})

describe("toolbar keys", () => {
  it("moves the focus along a vertical toolbar with the arrows, Home and End, skipping disabled buttons", () => {
    render(
      <div role="toolbar" aria-orientation="vertical" onKeyDown={toolbarKeys("vertical")}>
        <button type="button">Brush</button>
        <button type="button" disabled>
          Off
        </button>
        <button type="button">Eraser</button>
        <button type="button">Hand</button>
      </div>,
    )
    const [brush, eraser, hand] = ["Brush", "Eraser", "Hand"].map((name) => screen.getByRole("button", { name }))
    brush!.focus()
    fireEvent.keyDown(brush!, { key: "ArrowDown" })
    expect(eraser).toHaveFocus()
    fireEvent.keyDown(eraser!, { key: "End" })
    expect(hand).toHaveFocus()
    fireEvent.keyDown(hand!, { key: "ArrowDown" })
    expect(brush).toHaveFocus()
    fireEvent.keyDown(brush!, { key: "ArrowUp" })
    expect(hand).toHaveFocus()
    // The other axis is left alone.
    fireEvent.keyDown(hand!, { key: "ArrowRight" })
    expect(hand).toHaveFocus()
  })
})
