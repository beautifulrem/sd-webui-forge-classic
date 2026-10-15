import { act, fireEvent, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { resetMockNetworks } from "@/api/mock/networks"
import { queryClient } from "@/api/queries"
import { getFormStore } from "@/ui/form-store"
import { gridLayout, MAX_MOUNTED, moveIndex } from "./grid-layout"
import { DEFAULT_SURFACE, useNetworksUi } from "./store"
import { layout, listCalls, mockApi, renderBrowser, seedBootstrap, slow } from "./test-helpers"

const listbox = () => screen.findByRole("listbox", { name: "LoRA" }, slow)

describe("grid geometry", () => {
  it("never mounts more than MAX_MOUNTED cards when the viewport allows it", () => {
    for (const [width, height] of [
      [1600, 900],
      [900, 1200],
      [420, 800],
      [2400, 1300],
    ] as const) {
      for (const size of ["s", "m", "l"] as const) {
        const grid = gridLayout(width, height, "grid", size)
        const visibleRows = Math.ceil(height / grid.rowHeight) + 1
        const mounted = (visibleRows + 2 * grid.overscan) * grid.columns
        if (visibleRows * grid.columns <= MAX_MOUNTED) expect(mounted).toBeLessThanOrEqual(MAX_MOUNTED)
      }
    }
    const list = gridLayout(800, 900, "list", "m")
    expect(Math.ceil(900 / list.rowHeight) + 1 + 2 * list.overscan).toBeLessThanOrEqual(MAX_MOUNTED)
  })

  it("moves the cursor by keys", () => {
    expect(moveIndex("ArrowDown", 2, 100, 5, 3)).toBe(7)
    expect(moveIndex("ArrowUp", 2, 100, 5, 3)).toBe(0)
    expect(moveIndex("PageDown", 2, 100, 5, 3)).toBe(17)
    expect(moveIndex("End", 2, 100, 5, 3)).toBe(99)
    expect(moveIndex("ArrowRight", -1, 100, 5, 3)).toBe(0)
    expect(moveIndex("x", 2, 100, 5, 3)).toBeNull()
  })
})

describe("network browser", { timeout: 60_000 }, () => {
  beforeEach(() => {
    resetMockNetworks()
    queryClient.clear()
    seedBootstrap()
    useNetworksUi.setState({ surfaces: { page: { ...DEFAULT_SURFACE, tree: false }, drawer: DEFAULT_SURFACE }, target: null })
    getFormStore("txt2img").setState({ values: { core: { prompt: "1girl", negative_prompt: "" } } })
  })
  afterEach(() => vi.restoreAllMocks())

  it("virtualizes 5,000 LoRA: at most 60 cards are mounted", async () => {
    // (454 of them are in a hidden "." folder: listed only when searched)
    mockApi()
    layout(1200, 800)
    renderBrowser()
    const list = await listbox()
    await waitFor(() => expect(within(list).getAllByRole("option").length).toBeGreaterThan(0), slow)
    const mounted = within(list).getAllByRole("option").length
    expect(mounted).toBeLessThanOrEqual(MAX_MOUNTED)
    expect(screen.getByRole("searchbox", { name: "Search networks" }).parentElement).toHaveTextContent("4,546")
  })

  it("searches with every word", async () => {
    const calls = mockApi()
    layout(1200, 800)
    renderBrowser()
    await listbox()
    fireEvent.change(screen.getByRole("searchbox", { name: "Search networks" }), { target: { value: "watercolor_pastel V2_" } })
    await waitFor(() => {
      const labels = within(screen.getByRole("listbox", { name: "LoRA" }))
        .getAllByRole("option")
        .map((option) => option.getAttribute("aria-label") ?? "")
      expect(labels.length).toBeGreaterThan(0)
      expect(labels.every((label) => label.startsWith("watercolor_pastel_v2_"))).toBe(true)
    }, slow)
    expect(calls.some((call) => call.path === "/networks/lora")).toBe(true)
  })

  it("toggles a card in the focused form's prompt", async () => {
    mockApi()
    layout(1200, 800)
    renderBrowser()
    const list = await listbox()
    const [first] = await within(list).findAllByRole("option", {}, slow)
    const name = first!.getAttribute("aria-label")!.split(",")[0]!
    expect(first).toHaveAttribute("aria-selected", "false")
    fireEvent.click(first!)
    expect(getFormStore("txt2img").getState().values.core?.prompt).toBe(`1girl <lora:${name}:1>`)
    await waitFor(() => expect(within(list).getAllByRole("option")[0]).toHaveAttribute("aria-selected", "true"), slow)
    fireEvent.click(within(list).getAllByRole("option")[0]!)
    expect(getFormStore("txt2img").getState().values.core?.prompt).toBe("1girl")
  })

  it("sends cards to the negative prompt when that is the target", async () => {
    mockApi()
    layout(1200, 800)
    useNetworksUi.setState({ target: "negative" })
    renderBrowser()
    const list = await listbox()
    fireEvent.click((await within(list).findAllByRole("option", {}, slow))[0]!)
    expect(getFormStore("txt2img").getState().values.core?.negative_prompt).toMatch(/^<lora:/)
  })

  it("is driven by the keyboard and opens the metadata", async () => {
    mockApi()
    layout(1200, 800)
    renderBrowser()
    const list = await listbox()
    await within(list).findAllByRole("option", {}, slow)
    act(() => list.focus())
    fireEvent.keyDown(list, { key: "ArrowRight" })
    const options = within(list).getAllByRole("option")
    expect(list).toHaveAttribute("aria-activedescendant", options[1]!.id)
    fireEvent.keyDown(list, { key: "Enter" })
    expect(String(getFormStore("txt2img").getState().values.core?.prompt)).toContain("<lora:")
    fireEvent.keyDown(list, { key: "i" })
    const dialog = await screen.findByRole("dialog", {}, slow)
    expect(await within(dialog).findByText("Training tags", {}, slow)).toBeInTheDocument()
    expect(within(dialog).getByRole("button", { name: /^watercolor/ })).toBeInTheDocument()
  })

  it("loads a checkpoint from its card", async () => {
    const calls = mockApi()
    layout(1200, 800)
    useNetworksUi.setState({ surfaces: { page: { ...DEFAULT_SURFACE, kind: "checkpoints", tree: false }, drawer: DEFAULT_SURFACE } })
    renderBrowser()
    const list = await screen.findByRole("listbox", { name: "Checkpoints" }, slow)
    const option = (await within(list).findAllByRole("option", {}, slow))[0]!
    fireEvent.click(option)
    await waitFor(() => expect(calls.find((call) => call.method === "PATCH" && call.path === "/options")?.body).toEqual({ sd_model_checkpoint: expect.stringMatching(/\.safetensors \[/) }), slow)
    await waitFor(() => expect(within(list).getAllByRole("option")[0]).toHaveAttribute("aria-selected", "true"), slow)
  })

  it("filters by architecture with counted chips", async () => {
    mockApi()
    layout(1200, 800)
    renderBrowser()
    await listbox()
    const filters = await screen.findByRole("group", { name: "Filters" }, slow)
    const sdxl = await within(filters).findByRole("button", { name: /SDXL/ }, slow)
    expect(sdxl).toHaveAttribute("aria-pressed", "false")
    fireEvent.click(sdxl)
    expect(useNetworksUi.getState().surfaces.page.archs.lora).toEqual(["xl"])
    await waitFor(() => {
      const labels = within(screen.getByRole("listbox", { name: "LoRA" }))
        .getAllByRole("option")
        .map((option) => option.getAttribute("aria-label") ?? "")
      expect(labels.every((label) => label.includes("SDXL"))).toBe(true)
    }, slow)
  })
})

describe("network browser: windows, filters, errors", { timeout: 60_000 }, () => {
  beforeEach(() => {
    resetMockNetworks()
    queryClient.clear()
    useNetworksUi.setState({ surfaces: { page: { ...DEFAULT_SURFACE, tree: false }, drawer: DEFAULT_SURFACE }, target: null })
    getFormStore("txt2img").setState({ values: { core: { prompt: "", negative_prompt: "" } } })
  })
  afterEach(() => vi.restoreAllMocks())

  it("jumps to the end with one window, not every page before it", async () => {
    seedBootstrap()
    const calls = mockApi()
    layout(1200, 800)
    renderBrowser()
    const list = await listbox()
    await within(list).findAllByRole("option", {}, slow)
    // a scrollbar drag to the bottom (jsdom does not scroll by itself)
    Object.defineProperty(list, "scrollTop", { configurable: true, value: 10_000_000 })
    fireEvent.scroll(list)
    await waitFor(() => expect(within(list).getAllByRole("option").some((option) => option.id.endsWith("-o4545"))).toBe(true), slow)
    const offsets = new Set(listCalls(calls).map((call) => Number(call.url.searchParams.get("offset") ?? 0)))
    expect(offsets.has(4500)).toBe(true)
    // the first window and the ones around the end: nowhere near all 46
    expect(offsets.size).toBeLessThanOrEqual(4)
  })

  it("starts a new search at the top with a fresh grid", async () => {
    seedBootstrap()
    mockApi()
    layout(1200, 800)
    renderBrowser()
    const before = await listbox()
    await within(before).findAllByRole("option", {}, slow)
    act(() => before.focus())
    fireEvent.keyDown(before, { key: "PageDown" })
    expect(before).toHaveAttribute("aria-activedescendant")
    fireEvent.change(screen.getByRole("searchbox", { name: "Search networks" }), { target: { value: "ink" } })
    await waitFor(() => expect(screen.getByRole("listbox", { name: "LoRA" })).not.toBe(before), slow)
    const after = screen.getByRole("listbox", { name: "LoRA" })
    expect(after.scrollTop).toBe(0)
    expect(after).not.toHaveAttribute("aria-activedescendant")
  })

  it("never shows another kind's cards while a kind loads", async () => {
    seedBootstrap()
    mockApi()
    layout(1200, 800)
    renderBrowser()
    await within(await listbox()).findAllByRole("option", {}, slow)
    fireEvent.click(screen.getByRole("tab", { name: "Checkpoints" }))
    expect(screen.queryByRole("listbox", { name: "LoRA" })).toBeNull()
    expect(screen.queryByRole("listbox", { name: "Checkpoints" })?.querySelector("[aria-label^=watercolor]") ?? null).toBeNull()
    await screen.findByRole("listbox", { name: "Checkpoints" }, slow)
  })

  it("counts every folder in the tree's All row while a folder is chosen", async () => {
    seedBootstrap()
    mockApi()
    layout(1200, 800)
    useNetworksUi.setState({ surfaces: { page: { ...DEFAULT_SURFACE, tree: false, folder: { lora: "poses" } }, drawer: DEFAULT_SURFACE } })
    renderBrowser()
    await listbox()
    fireEvent.click(screen.getByRole("button", { name: "Folders" }))
    const tree = await screen.findByRole("tree", { name: "Folders" }, slow)
    const all = within(tree).getAllByRole("treeitem")[0]!
    await waitFor(() => expect(all).toHaveTextContent("4,546"), slow)
  })

  it("starts LoRA with the compatible filter when lora_preset_filter is on", async () => {
    seedBootstrap({ lora_preset_filter: true })
    const calls = mockApi({ options: { lora_preset_filter: true } })
    layout(1200, 800)
    renderBrowser()
    await listbox()
    await waitFor(() => expect(listCalls(calls).at(-1)?.url.searchParams.get("compatible")).toBe("true"), slow)
    expect(await screen.findByRole("button", { name: /For / }, slow)).toHaveAttribute("aria-pressed", "true")
  })

  it("sorts as the settings say until a sort is chosen", async () => {
    seedBootstrap({ extra_networks_card_order_field: "Date Created", extra_networks_card_order: "Descending" })
    const calls = mockApi({ options: { extra_networks_card_order_field: "Date Created", extra_networks_card_order: "Descending" } })
    layout(1200, 800)
    renderBrowser()
    await listbox()
    const first = listCalls(calls)[0]!.url.searchParams
    expect([first.get("sort"), first.get("order")]).toEqual(["created", "desc"])
  })

  it("clears the filters without a request for the old search", async () => {
    seedBootstrap()
    const calls = mockApi()
    layout(1200, 800)
    renderBrowser()
    await listbox()
    fireEvent.change(screen.getByRole("searchbox", { name: "Search networks" }), { target: { value: "ink" } })
    await waitFor(() => expect(listCalls(calls).some((call) => call.url.searchParams.get("q") === "ink")).toBe(true), slow)
    const filters = await screen.findByRole("group", { name: "Filters" }, slow)
    const count = listCalls(calls).length
    fireEvent.click(within(filters).getByRole("button", { name: "Clear filters" }))
    await new Promise((resolve) => setTimeout(resolve, 400))
    const after = listCalls(calls).slice(count)
    expect(after.every((call) => !call.url.searchParams.get("q"))).toBe(true)
    expect(useNetworksUi.getState().surfaces.page.query).toBe("")
  })

  it("shows an error with a working Retry", async () => {
    seedBootstrap()
    let broken = true
    mockApi({ fail: (call) => broken && call.path === "/networks/lora" })
    layout(1200, 800)
    renderBrowser()
    const alert = await screen.findByRole("alert", {}, slow)
    expect(alert).toHaveTextContent("Could not load LoRA")
    broken = false
    fireEvent.click(within(alert).getByRole("button", { name: "Retry" }))
    await within(await listbox()).findAllByRole("option", {}, slow)
  })

  it("does not pretend there are no networks when the kinds fail to load", async () => {
    seedBootstrap()
    mockApi({ fail: (call) => call.path === "/networks" })
    layout(1200, 800)
    renderBrowser()
    expect(await screen.findByRole("alert", {}, slow)).toHaveTextContent("Could not load the network kinds")
    expect(screen.queryByText("Put files into the model folder, then refresh.")).toBeNull()
  })

  it("offers Open folder only when the server allows it", async () => {
    seedBootstrap({}, [])
    mockApi({ features: [] })
    layout(1200, 800)
    renderBrowser()
    const list = await listbox()
    await within(list).findAllByRole("option", {}, slow)
    act(() => list.focus())
    fireEvent.keyDown(list, { key: "ArrowRight" })
    fireEvent.keyDown(list, { key: "." })
    const menu = await screen.findByRole("menu", {}, slow)
    expect(within(menu).getByRole("menuitem", { name: /Copy path/ })).toBeInTheDocument()
    expect(within(menu).queryByRole("menuitem", { name: /Open folder/ })).toBeNull()
  })

  it("keeps the card's buttons out of the listbox's accessibility tree", async () => {
    seedBootstrap()
    mockApi()
    layout(1200, 800)
    renderBrowser()
    const list = await listbox()
    const [option] = await within(list).findAllByRole("option", {}, slow)
    expect(option!.querySelector("button")).toBeNull()
    expect(within(list).queryAllByRole("button")).toHaveLength(0)
    expect(list.querySelector("[aria-live]")).toBeNull()
  })

  it("saves the editor's fields as a PATCH of the sidecar", async () => {
    seedBootstrap()
    const calls = mockApi()
    layout(1200, 800)
    renderBrowser()
    const list = await listbox()
    await within(list).findAllByRole("option", {}, slow)
    act(() => list.focus())
    fireEvent.keyDown(list, { key: "ArrowRight" })
    fireEvent.keyDown(list, { key: "e" })
    const dialog = await screen.findByRole("dialog", {}, slow)
    fireEvent.change(await within(dialog).findByRole("textbox", { name: "Activation text" }, slow), { target: { value: " watercolor, soft " } })
    fireEvent.change(within(dialog).getByRole("textbox", { name: "Tags" }), { target: { value: "paint, , style" } })
    fireEvent.click(within(dialog).getByRole("button", { name: "Save" }))
    await waitFor(() => expect(calls.find((call) => call.method === "PATCH")).toBeTruthy(), slow)
    const patch = calls.find((call) => call.method === "PATCH")!
    expect(patch.path).toMatch(/^\/networks\/lora\/.+\/meta$/)
    expect(patch.body).toEqual({
      user_metadata: { description: null, "sd version": null, "activation text": "watercolor, soft", "preferred weight": null, "negative text": null, tags: ["paint", "style"], notes: null },
      replace: false,
    })
  })
})
