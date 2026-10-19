import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { SearchSelect } from "./SearchSelect"

const options = ["Euler a", "Euler", "DPM++ 2M"].map((value) => ({ value, label: value }))

describe("SearchSelect", () => {
  it("picks the exact match with Enter, before longer matches listed earlier", async () => {
    const user = userEvent.setup()
    const onValueChange = vi.fn()
    render(<SearchSelect options={options} value="DPM++ 2M" onValueChange={onValueChange} aria-label="Sampler" />)
    await user.click(screen.getByRole("combobox", { name: "Sampler" }))
    await user.click(await screen.findByPlaceholderText("Search…"))
    await user.keyboard("euler{Enter}")
    await waitFor(() => expect(onValueChange).toHaveBeenCalledWith("Euler"))
    expect(onValueChange).toHaveBeenCalledTimes(1)
  })

  it("picks the first match with Enter while filtering", async () => {
    const user = userEvent.setup()
    const onValueChange = vi.fn()
    render(<SearchSelect options={options} value="Euler" onValueChange={onValueChange} aria-label="Sampler" />)
    await user.click(screen.getByRole("combobox", { name: "Sampler" }))
    await user.click(await screen.findByPlaceholderText("Search…"))
    await user.keyboard("dpm{Enter}")
    await waitFor(() => expect(onValueChange).toHaveBeenCalledWith("DPM++ 2M"))
  })

  it("names its popup dialog after the trigger's label", async () => {
    const user = userEvent.setup()
    render(
      <>
        <label htmlFor="sampler">Sampling method</label>
        <SearchSelect id="sampler" options={options} value="Euler" onValueChange={() => {}} />
      </>,
    )
    await user.click(screen.getByRole("combobox", { name: "Sampling method" }))
    expect(await screen.findByRole("dialog", { name: "Sampling method" })).toBeInTheDocument()
  })

  it("names its popup dialog after an aria-label", async () => {
    const user = userEvent.setup()
    render(<SearchSelect options={options} value="Euler" onValueChange={() => {}} aria-label="Checkpoint" />)
    await user.click(screen.getByRole("combobox", { name: "Checkpoint" }))
    expect(await screen.findByRole("dialog", { name: "Checkpoint" })).toBeInTheDocument()
  })
})
