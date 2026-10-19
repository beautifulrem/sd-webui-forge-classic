import { act, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"
import { i18n } from "@/i18n"
import { NumberField } from "./ui/number-field"

describe("NumberField", () => {
  afterEach(async () => {
    await act(() => i18n.changeLanguage("en"))
  })

  it("names its stepper buttons in the current language", async () => {
    i18n.addResourceBundle("zh-CN", "chrome", { Increase: "增加", Decrease: "减少" }, true, true)
    render(<NumberField aria-label="Steps" defaultValue={20} />)
    expect(screen.getByRole("button", { name: "Increase" })).toBeInTheDocument()
    await act(() => i18n.changeLanguage("zh-CN"))
    expect(screen.getByRole("button", { name: "增加" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "减少" })).toBeInTheDocument()
  })
})
