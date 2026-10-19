/**
 * The settings page against the real settings schema
 * (`api/mock/contract/settings.json`, `tools/dump_settings.py`): it indexes,
 * searches, renders every widget kind, applies edits and pins settings.
 */
import { QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import settingsContract from "@/api/mock/contract/settings.json"
import { bootstrap } from "@/api/mock/fixtures"
import { keys, queryClient } from "@/api/queries"
import type { SettingsSchema } from "@/ui/schema"
import SettingsPage from "./index"
import { INITIAL_FILTERS, useSettingsFilters } from "./settings-filters"
import { buildSettingsIndex, filterSettings, settingInputs } from "./settings-model"
import { useSettingsEditor } from "./settings-store"

const contract = settingsContract as unknown as SettingsSchema

function page(query = "") {
  return (
    <QueryClientProvider client={queryClient}>
      <SettingsPage route={{ name: "settings", query: query ? new URLSearchParams(query) : undefined }} />
    </QueryClientProvider>
  )
}

function renderPage(query = "") {
  return render(page(query))
}

describe("real settings schema", () => {
  it("indexes every category and finds settings by key, label and help", () => {
    const index = buildSettingsIndex(contract)
    expect(index.sections.length).toBeGreaterThan(20)
    expect(settingInputs(contract).length).toBeGreaterThan(200)
    const find = (query: string) => filterSettings(index, { query, mode: "all", values: contract.values, saved: contract.values, defaults: contract.defaults, translate: (s) => s })
    const keysFor = (query: string) => find(query).flatMap((s) => s.visible.flatMap((i) => (i.kind === "setting" ? [i.key] : i.kind === "row" ? i.keys : [])))
    expect(keysFor("jpeg_quality")).toContain("jpeg_quality")
    expect(keysFor("live preview")).toContain("live_previews_enable")
    // choice descriptions are searchable help
    expect(keysFor("high quality preview method")).toContain("show_progress_type")
    expect(Object.keys(contract.defaults ?? {})).toEqual(Object.keys(contract.values))
  })
})

describe("settings page", { timeout: 20_000 }, () => {
  let patch: ReturnType<typeof vi.spyOn>
  beforeEach(() => {
    // jsdom has no layout: give the virtualized list a viewport.
    vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockReturnValue(800)
    vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockReturnValue(1000)
    useSettingsEditor.setState({ schema: null, saved: {}, inputs: new Map(), applying: false, notice: { restart: [], reload: [] } })
    useSettingsFilters.setState(INITIAL_FILTERS)
    queryClient.setQueryData(keys.app, structuredClone(bootstrap))
    vi.spyOn(api, "options").mockResolvedValue(structuredClone(contract))
    vi.spyOn(api, "app").mockResolvedValue(structuredClone(bootstrap))
    vi.spyOn(api.system, "status").mockResolvedValue({ restartable: true, extension_access: true, settings_frozen: false, model_loaded: null, version: "test" })
    patch = vi.spyOn(api, "patchOptions").mockImplementation(async (changes) => ({ ...structuredClone(contract), values: { ...contract.values, ...(changes as SettingsSchema["values"]) } }))
  })
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("renders the nav and the first settings", async () => {
    renderPage()
    const nav = await screen.findByRole("navigation", { name: "Settings sections" }, { timeout: 8000 })
    expect(within(nav).getByRole("button", { name: "Licenses" })).toBeInTheDocument()
    await waitFor(() => expect(within(nav).getAllByRole("button").length).toBeGreaterThan(20))
  })

  it("searches, applies an edit with Ctrl+S and shows it as saved", async () => {
    renderPage()
    const search = await screen.findByRole("searchbox", { name: "Search settings" }, { timeout: 8000 })
    fireEvent.change(search, { target: { value: "samples_save" } })
    const toggle = await screen.findByRole("switch", { name: "Automatically save every generated image" }, { timeout: 8000 })
    const before = toggle.getAttribute("aria-checked") === "true"
    fireEvent.click(toggle)
    expect(await screen.findByText("Unsaved changes: 1")).toBeInTheDocument()
    await act(async () => {
      fireEvent.keyDown(window, { key: "s", ctrlKey: true })
    })
    await waitFor(() => expect(patch).toHaveBeenCalledWith({ samples_save: !before }))
    await waitFor(() => expect(screen.queryByText("Unsaved changes: 1")).not.toBeInTheDocument())
  })

  it("shows nothing-found with a way out, and filters to unsaved edits", async () => {
    renderPage()
    const search = await screen.findByRole("searchbox", { name: "Search settings" }, { timeout: 8000 })
    fireEvent.change(search, { target: { value: "zzz-no-such-setting" } })
    expect(await screen.findByText("No settings found")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Clear filters" }))
    await waitFor(() => expect(screen.queryByText("No settings found")).not.toBeInTheDocument())
    fireEvent.click(screen.getByRole("radio", { name: "Unsaved" }))
    expect(await screen.findByText("No settings found")).toBeInTheDocument()
  })

  it("opens a system view from the query", async () => {
    vi.spyOn(api.system, "licenses").mockResolvedValue({ groups: [{ title: "Special Thanks", items: [{ id: "x", name: "ComfyUI", url: "https://github.com/x", description: "Code", license_url: null }] }] })
    renderPage("view=licenses")
    // A lazy view: allow for its chunk to load.
    expect(await screen.findByRole("heading", { name: "Licenses" }, { timeout: 8000 })).toBeInTheDocument()
    expect(await screen.findByText("ComfyUI", {}, { timeout: 8000 })).toBeInTheDocument()
  })

  it("links the API documentation and the loaded checkpoint's hash (the old footer)", async () => {
    const hash = "0123456789abcdef".repeat(4)
    vi.spyOn(api.system, "status").mockResolvedValue({ restartable: true, extension_access: true, settings_frozen: false, model_loaded: "anima.safetensors [0123456789]", model_hash: hash, version: "test" })
    renderPage("view=server")
    const docs = await screen.findByRole("link", { name: "API documentation" }, { timeout: 8000 })
    expect(docs).toHaveAttribute("href", "/docs")
    const civitai = await screen.findByRole("link", { name: "Search on CivitAI" })
    expect(civitai).toHaveAttribute("href", `https://civitai.com/search/models?query=${hash}`)
    expect(civitai).toHaveAttribute("rel", "noopener noreferrer")
    expect(screen.getByText("0123456789", { selector: "code" })).toHaveAttribute("title", hash)
  })

  it("keeps the search when the page is left, and follows a search from the route", async () => {
    const first = renderPage()
    const search = await screen.findByRole("searchbox", { name: "Search settings" }, { timeout: 8000 })
    fireEvent.change(search, { target: { value: "jpeg" } })
    first.unmount()
    const second = renderPage()
    expect(await screen.findByRole("searchbox", { name: "Search settings" }, { timeout: 8000 })).toHaveValue("jpeg")
    // A link with another search (or filter) wins over the remembered one.
    second.rerender(page("q=webp&filter=changed"))
    await waitFor(() => expect(screen.getByRole("searchbox", { name: "Search settings" })).toHaveValue("webp"))
    expect(screen.getByRole("radio", { name: "Changed" })).toBeChecked()
  })

  it("describes choices and an ordering setting's default order as text, not server markup", async () => {
    const ordered = structuredClone(contract)
    const section = ordered.categories[0]!.sections[0]!
    section.items.push({ type: "multichoice", name: "prioritized_callbacks_demo", label: "demo callback priority", default: [], choices: [{ value: "b", label: "b" }, { value: "a", label: "a" }], default_order: ["b", "a"], needs_restart: true })
    ordered.values.prioritized_callbacks_demo = []
    vi.spyOn(api, "options").mockResolvedValue(ordered)
    const view = renderPage("q=show_progress_type")
    const row = (await screen.findByText("show_progress_type", { selector: "code" }, { timeout: 8000 })).closest("[data-setting]")!
    const help = within(row as HTMLElement)
    expect(help.getByText("legacy preview method", { exact: false }).closest("li")).toHaveTextContent("Approx NN: legacy preview method")
    expect(help.getByText("fast but low quality preview method", { exact: false }).closest("li")).toHaveTextContent("RGB: fast but low quality preview method")
    expect(row.querySelector(".info, ul[style], li[style]")).toBeNull()
    view.unmount()
    renderPage("q=prioritized_callbacks_demo")
    const orderRow = (await screen.findByText("prioritized_callbacks_demo", { selector: "code" }, { timeout: 8000 })).closest("[data-setting]")!
    expect(within(orderRow as HTMLElement).getByText("Default order:")).toBeInTheDocument()
    expect([...orderRow.querySelectorAll("ol li")].map((item) => item.textContent)).toEqual(["b", "a"])
  })

  it("disables settings the server refuses, with the reason", async () => {
    const locked = structuredClone(contract)
    for (const category of locked.categories)
      for (const section of category.sections)
        for (const item of section.items)
          if ("name" in item && item.name === "jpeg_quality") Object.assign(item, { enabled_if: false, locked: "this setting is frozen" })
    vi.spyOn(api, "options").mockResolvedValue(locked)
    renderPage("q=jpeg_quality")
    expect(await screen.findByText(/this setting is frozen/, {}, { timeout: 8000 })).toBeInTheDocument()
    const row = screen.getByText("jpeg_quality", { selector: "code" }).closest("[data-setting]")!
    for (const input of row.querySelectorAll("input")) expect(input).toBeDisabled()
    expect(row.querySelectorAll("input").length).toBeGreaterThan(0)
  })
})

describe("settings navigation while loading", () => {
  it("shows placeholders, not the empty-search message, until the schema arrives", async () => {
    const { SettingsNav } = await import("./SettingsNav")
    const props = { view: "settings" as const, activeSection: null, filtering: false, unsaved: new Map<string, number>(), onSection: () => {}, onView: () => {} }
    const { rerender } = render(<SettingsNav sections={[]} loading {...props} />)
    expect(screen.queryByText("No matching sections")).toBeNull()
    expect(screen.getByRole("status")).toHaveAttribute("aria-busy", "true")
    rerender(<SettingsNav sections={[]} {...props} />)
    expect(screen.getByText("No matching sections")).toBeVisible()
  })
})
