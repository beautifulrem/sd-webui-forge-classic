import { render } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { PageSkeleton } from "./PageSkeleton"

describe("PageSkeleton", () => {
  it("marks the page as loading (the e2e helpers and assistive technology wait on aria-busy)", () => {
    const { container } = render(<PageSkeleton />)
    const root = container.firstElementChild!
    expect(root).toHaveAttribute("aria-busy", "true")
    // Placeholders for the parameters column and the stage.
    expect(root.querySelectorAll('[data-slot="skeleton"]').length).toBeGreaterThanOrEqual(6)
  })
})
