/**
 * The search select's stand-in (SearchSelect.tsx) before its combobox code has loaded: the closed
 * select, then the focus handed to the real trigger. A file of its own: once loaded, the
 * implementation stays loaded for the rest of a test file.
 */
import { act, render, screen, waitFor } from "@testing-library/react"
import { expect, it } from "vitest"
import { SearchSelect } from "./SearchSelect"

const options = ["Euler a", "Euler", "DPM++ 2M"].map((value) => ({ value, label: value }))

it("shows the closed select, then hands the focus to the real trigger", { timeout: 30_000 }, async () => {
  render(<SearchSelect options={options} value="Euler" onValueChange={() => {}} aria-label="Sampler" suffix="3 s" />)
  const standIn = screen.getByRole("combobox", { name: "Sampler" })
  expect(standIn).toHaveAttribute("aria-haspopup", "dialog")
  expect(standIn).toHaveAttribute("aria-expanded", "false")
  expect(standIn).toHaveAttribute("title", "Euler")
  expect(standIn).toHaveTextContent("Euler3 s")
  act(() => standIn.focus())
  await waitFor(() => expect(screen.getByRole("combobox", { name: "Sampler" })).not.toBe(standIn), { timeout: 15_000 })
  const trigger = screen.getByRole("combobox", { name: "Sampler" })
  expect(trigger).toHaveFocus()
  expect(trigger).toHaveTextContent("Euler3 s")
  expect(screen.queryByRole("dialog")).toBeNull()
})
