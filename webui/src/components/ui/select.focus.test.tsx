/** The select's stand-in hands the focus to the real trigger that replaces it (select.tsx). */
import { act, render, screen, waitFor } from "@testing-library/react"
import { expect, it } from "vitest"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./select"

const items = [
  { value: "euler", label: "Euler" },
  { value: "dpm", label: "DPM++ 2M" },
]

it("moves the focus to the real trigger when it replaces a focused stand-in", { timeout: 30_000 }, async () => {
  render(
    <Select<string> items={items} value="dpm">
      <SelectTrigger aria-label="Sampler">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {items.map((item) => (
          <SelectItem key={item.value} value={item.value}>
            {item.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>,
  )
  const standIn = screen.getByRole("combobox", { name: "Sampler" })
  act(() => standIn.focus())
  // the focus fetched the implementation: the stand-in is replaced and the focus follows
  await waitFor(() => expect(screen.getByRole("combobox", { name: "Sampler" })).not.toBe(standIn), { timeout: 15_000 })
  const trigger = screen.getByRole("combobox", { name: "Sampler" })
  expect(trigger).toHaveFocus()
  expect(trigger).toHaveTextContent("DPM++ 2M")
  expect(screen.queryByRole("listbox")).toBeNull()
})
