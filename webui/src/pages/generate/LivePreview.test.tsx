import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { LivePreview } from "./LivePreview"

// whether the frame animates in (it must not: one drawn frame per generation step)
const frames = (container: HTMLElement) => [...container.querySelectorAll("img")].map((img) => [img.getAttribute("src"), img.dataset.frame, /animate|fade|transition/.test(img.className)])

describe("live preview", () => {
  it("stacks a new frame over the previous one without animating it, keeping two at most", () => {
    const { container, rerender } = render(<LivePreview src="blob:1" alt="Live preview" />)
    expect(frames(container)).toEqual([["blob:1", "current", false]])

    rerender(<LivePreview src="blob:2" alt="Live preview" />)
    expect(frames(container)).toEqual([
      ["blob:1", "previous", false],
      ["blob:2", "current", false],
    ])
    // only the frame on top is the image; the one underneath is decorative
    expect(screen.getByRole("img", { name: "Live preview" })).toHaveAttribute("src", "blob:2")
    expect(container.querySelector("[data-frame=previous]")).toHaveAttribute("aria-hidden", "true")

    const second = container.querySelector("[data-frame=current]")
    rerender(<LivePreview src="blob:3" alt="Live preview" />)
    expect(frames(container)).toEqual([
      ["blob:2", "previous", false],
      ["blob:3", "current", false],
    ])
    // the element that showed frame 2 stays mounted (no reload, no blank frame)
    expect(container.querySelector("[data-frame=previous]")).toBe(second)
  })

  it("keeps its frames when re-rendered with the same frame", () => {
    const { container, rerender } = render(<LivePreview src="blob:1" alt="Live preview" />)
    rerender(<LivePreview src="blob:1" alt="Live preview" />)
    expect(frames(container)).toEqual([["blob:1", "current", false]])
  })
})
