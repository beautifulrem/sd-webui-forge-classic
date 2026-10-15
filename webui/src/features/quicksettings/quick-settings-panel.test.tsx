/** The quick settings popover content: pinned settings rendered by the schema renderer, applied as they change. */
import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { bootstrap } from "@/api/mock/fixtures"
import { keys, queryClient } from "@/api/queries"
import type { SettingsSchema } from "@/ui/schema"
import QuickSettingsPanel from "./QuickSettingsPanel"

const schema: SettingsSchema = {
  version: 1,
  categories: [
    {
      id: "c",
      label: "C",
      sections: [
        {
          id: "s",
          label: "S",
          items: [
            { type: "toggle", name: "samples_save", label: "Save every image", default: true },
            { type: "toggle", name: "grid_save", label: "Save grids", default: true, enabled_if: false, locked: "this setting is frozen" },
          ],
        },
      ],
    },
  ],
  quicksettings: ["samples_save", "grid_save", "gone_setting"],
  values: { samples_save: true, grid_save: true, quicksettings_list: ["samples_save", "grid_save", "gone_setting"] },
  defaults: { samples_save: true, grid_save: true },
}

function renderPanel() {
  return render(
    <QueryClientProvider client={queryClient}>
      <QuickSettingsPanel keys={["samples_save", "grid_save", "gone_setting"]} />
    </QueryClientProvider>,
  )
}

describe("quick settings panel", () => {
  beforeEach(() => {
    queryClient.setQueryData(keys.app, { ...structuredClone(bootstrap), options: { ...bootstrap.options, quicksettings_list: ["samples_save", "grid_save", "gone_setting"] } })
    vi.spyOn(api, "options").mockResolvedValue(structuredClone(schema))
    vi.spyOn(api, "app").mockResolvedValue(structuredClone(bootstrap))
  })
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("renders the pinned settings, writes a change at once and keeps locked ones disabled", async () => {
    const patch = vi.spyOn(api, "patchOptions").mockImplementation(async (changes) => ({ ...structuredClone(schema), values: { ...schema.values, ...(changes as SettingsSchema["values"]) } }))
    renderPanel()
    const save = await screen.findByRole("switch", { name: "Save every image" })
    expect(screen.getByRole("switch", { name: "Save grids" })).toHaveAttribute("aria-disabled", "true")
    fireEvent.click(save)
    await waitFor(() => expect(patch).toHaveBeenCalledWith({ samples_save: false }))
  })

  it("lists pins the server does not know and unpins them", async () => {
    const patch = vi.spyOn(api, "patchOptions").mockResolvedValue(structuredClone(schema))
    renderPanel()
    expect(await screen.findByText("gone_setting")).toBeInTheDocument()
    const unpin = screen.getAllByRole("button", { name: "Unpin" }).at(-1)!
    fireEvent.click(unpin)
    await waitFor(() => expect(patch).toHaveBeenCalledWith({ quicksettings_list: ["samples_save", "grid_save"] }))
  })
})
