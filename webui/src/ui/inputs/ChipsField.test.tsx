/**
 * A dropdown-style multiple choice renders a stand-in of its chips field until it is used
 * (MultiChoice.tsx, ChipsField); its combobox code (ChipsSelect.tsx) then takes over. Once
 * loaded it stays loaded for the rest of this file: the stand-in is checked first.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { createFormStore, type FormStore } from "../form-store"
import type { TabSchema } from "../schema"
import { TabForm } from "../SchemaForm"

const choices = [
  { value: "x", label: "Ex" },
  { value: "y", label: "Why" },
  { value: "z", label: "Zed" },
]

function renderChips(): FormStore {
  const schema: TabSchema = {
    version: 1,
    tab: "txt2img",
    core: { owner: "core", layout: { type: "column", children: [{ type: "multichoice", name: "styles", label: "Styles", choices }] }, defaults: { styles: ["y"] } },
    scripts: [],
  }
  const store = createFormStore(`chips-${Math.random()}`, { persistDrafts: false })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, enabled: false } } })
  render(
    <QueryClientProvider client={client}>
      <TabForm schema={schema} formId="txt2img" store={store} />
    </QueryClientProvider>,
  )
  return store
}

const slow = { timeout: 15_000 }

describe("chips field (combobox loaded on demand)", { timeout: 30_000 }, () => {
  it("shows the chips and a search field, removes a chip, then carries typed text into the real field", async () => {
    const store = renderChips()
    const field = screen.getByRole("combobox", { name: "Styles" })
    expect(field).toHaveAttribute("aria-expanded", "false")
    expect(screen.getByText("Why", { selector: "[data-slot=combobox-chip]" })).toBeInTheDocument()
    // a chip's remove button works before the combobox code is there
    fireEvent.click(within(screen.getByText("Why", { selector: "[data-slot=combobox-chip]" })).getByRole("button"))
    expect(store.getState().values.core?.styles).toEqual([])

    // typing into the stand-in: the real field takes over, opened, with the text and the focus
    act(() => field.focus())
    fireEvent.change(field, { target: { value: "Ze" } })
    const listbox = await screen.findByRole("listbox", {}, slow)
    const input = screen.getByRole("combobox", { name: "Styles" })
    expect(input).toHaveValue("Ze")
    await waitFor(() => expect(input).toHaveFocus())
    expect(within(listbox).getByRole("option", { name: "Zed" })).toBeInTheDocument()
    expect(within(listbox).queryByRole("option", { name: "Ex" })).toBeNull()
  })
})
