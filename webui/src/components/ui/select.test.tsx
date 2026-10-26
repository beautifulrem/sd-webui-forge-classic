/**
 * The select renders a stand-in of its closed state until its implementation (Base UI Select) has
 * loaded. The implementation stays loaded for the rest of a test file: the stand-in tests come
 * first, the focus hand-over has a file of its own (select.focus.test.tsx).
 */
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./select"

const items = [
  { value: "euler", label: "Euler" },
  { value: "dpm", label: "DPM++ 2M" },
]

const slow = { timeout: 15_000 }

describe("select (implementation loaded on demand)", { timeout: 30_000 }, () => {
  function Choice({ value = "dpm", onValueChange = () => {} }: { value?: string | null; onValueChange?: (value: string | null) => void }) {
    return (
      <>
        <label htmlFor="sampler">Sampler</label>
        <Select<string> items={items} value={value} onValueChange={onValueChange}>
          <SelectTrigger id="sampler">
            <SelectValue placeholder="Pick one" />
          </SelectTrigger>
          <SelectContent>
            {items.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                {item.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </>
    )
  }

  it("shows the closed select with the selected item's label before its code has loaded", () => {
    render(<Choice />)
    const trigger = screen.getByRole("combobox", { name: "Sampler" })
    expect(trigger).toHaveAttribute("aria-haspopup", "listbox")
    expect(trigger).toHaveAttribute("aria-expanded", "false")
    expect(trigger).toHaveAttribute("data-slot", "select-trigger")
    expect(trigger).toHaveTextContent("DPM++ 2M")
    expect(screen.queryByRole("listbox")).toBeNull()
  })

  it("shows the placeholder without a value", () => {
    render(<Choice value={null} />)
    const trigger = screen.getByRole("combobox", { name: "Sampler" })
    expect(trigger).toHaveTextContent("Pick one")
    expect(trigger).toHaveAttribute("data-placeholder")
  })

  it("opens on a press (not replaced under the pointer by the press's focus) and picks an option", async () => {
    const onValueChange = vi.fn()
    render(<Choice onValueChange={onValueChange} />)
    const standIn = screen.getByRole("combobox", { name: "Sampler" })
    // the press focuses the stand-in first: it stays under the pointer (the click must reach it)
    fireEvent.pointerDown(standIn, { pointerType: "mouse" })
    act(() => standIn.focus())
    await act(() => import("./select-impl"))
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(standIn).toBeInTheDocument()
    fireEvent.click(standIn)
    const listbox = await screen.findByRole("listbox", {}, slow)
    // a mouse press on the option (Base UI selects on a click that started on the item)
    const option = within(listbox).getByRole("option", { name: "Euler" })
    fireEvent.pointerDown(option, { pointerType: "mouse" })
    fireEvent.click(option)
    await waitFor(() => expect(onValueChange).toHaveBeenCalledWith("euler", expect.anything()))
  })
})
