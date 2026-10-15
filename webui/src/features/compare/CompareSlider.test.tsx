import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { CompareSlider } from "./CompareSlider"

describe("compare slider", () => {
  it("moves the divider from the keyboard and clips the after image", () => {
    render(<CompareSlider before="/a.png" after="/b.png" beforeLabel="Source" afterLabel="Result" aspect={1} />)
    const slider = screen.getByRole("slider", { name: "Comparison divider" })
    expect(slider).toHaveAttribute("aria-valuenow", "50")
    fireEvent.keyDown(slider, { key: "ArrowRight", shiftKey: true })
    expect(slider).toHaveAttribute("aria-valuenow", "60")
    expect(screen.getByRole("img", { name: "Result" }).style.clipPath).toBe("inset(0 0 0 60.0%)")
    fireEvent.keyDown(slider, { key: "Home" })
    expect(slider).toHaveAttribute("aria-valuenow", "0")
  })

  it("follows the pointer across the frame", () => {
    render(<CompareSlider before="/a.png" after="/b.png" aspect={2} />)
    const frame = screen.getByRole("img", { name: "Before" }).parentElement!
    vi.spyOn(frame, "getBoundingClientRect").mockReturnValue({ left: 100, width: 200, top: 0, height: 100, right: 300, bottom: 100, x: 100, y: 0, toJSON: () => ({}) })
    fireEvent.pointerDown(frame, { clientX: 250, button: 0, pointerId: 1 })
    expect(screen.getByRole("slider")).toHaveAttribute("aria-valuenow", "75")
    fireEvent.pointerMove(frame, { clientX: 120, pointerId: 1 })
    expect(screen.getByRole("slider")).toHaveAttribute("aria-valuenow", "10")
    fireEvent.pointerUp(frame, { pointerId: 1 })
    fireEvent.pointerMove(frame, { clientX: 300, pointerId: 1 })
    expect(screen.getByRole("slider")).toHaveAttribute("aria-valuenow", "10")
  })
})
