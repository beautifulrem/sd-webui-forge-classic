import { act, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"
import { i18n } from "@/i18n"
import { InfotextPanel } from "./InfotextPanel"

describe("InfotextPanel", () => {
  afterEach(async () => {
    await act(() => i18n.changeLanguage("en"))
  })

  it("shows known parameter names in the current language, the raw key as a tooltip", async () => {
    i18n.addResourceBundle("zh-CN", "server", { Steps: "步数", Sampler: "采样器" }, true, true)
    await act(() => i18n.changeLanguage("zh-CN"))
    render(<InfotextPanel infotext={"a fox\nSteps: 20, Sampler: Euler a, Custom key: 1"} />)
    expect(screen.getByText("步数")).toHaveAttribute("title", "Steps")
    expect(screen.getByText("采样器")).toBeInTheDocument()
    expect(screen.getByText("Custom key")).not.toHaveAttribute("title") // unknown keys stay as they are
    expect(screen.getByText("Euler a")).toBeInTheDocument()
  })
})
