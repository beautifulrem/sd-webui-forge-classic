import { QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { resetMockNetworks } from "@/api/mock/networks"
import { queryClient } from "@/api/queries"
import NetworksDrawer from "./NetworksDrawer"
import { NetworksDrawerHost } from "./NetworksDrawerHost"
import { DRAWER_WIDTH, DEFAULT_SURFACE, useNetworksUi } from "./store"
import { layout, mockApi, seedBootstrap, slow } from "./test-helpers"

/** A desktop window (the drawer is a side panel with a resize handle). */
function desktop() {
  vi.spyOn(window, "matchMedia").mockImplementation(
    (query: string) =>
      ({ matches: !query.includes("reduce"), media: query, onchange: null, addEventListener: () => {}, removeEventListener: () => {}, addListener: () => {}, removeListener: () => {}, dispatchEvent: () => false }) as MediaQueryList,
  )
}

function renderDrawer() {
  const onClose = vi.fn()
  const returnFocus = vi.fn()
  render(
    <QueryClientProvider client={queryClient}>
      <NetworksDrawer tab="txt2img" autoFocus onClose={onClose} returnFocus={returnFocus} />
    </QueryClientProvider>,
  )
  return { panel: screen.getByRole("dialog", { name: "Networks" }), onClose, returnFocus }
}

describe("networks drawer", { timeout: 60_000 }, () => {
  beforeEach(() => {
    resetMockNetworks()
    queryClient.clear()
    seedBootstrap()
    useNetworksUi.setState({ surfaces: { page: DEFAULT_SURFACE, drawer: { ...DEFAULT_SURFACE, tree: false } }, drawerOpen: false, drawerWidth: DRAWER_WIDTH.default, target: null })
  })
  afterEach(() => vi.restoreAllMocks())

  it("closes on Escape and hands the focus back", async () => {
    mockApi()
    layout(900, 700)
    const { panel, onClose, returnFocus } = renderDrawer()
    const search = await screen.findByRole("searchbox", { name: "Search networks" })
    await waitFor(() => expect(document.activeElement).toBe(search))
    // Escape first clears the search text
    fireEvent.change(search, { target: { value: "ink" } })
    fireEvent.keyDown(search, { key: "Escape" })
    expect(onClose).not.toHaveBeenCalled()
    expect(search).toHaveValue("")
    fireEvent.keyDown(panel, { key: "Escape" })
    expect(onClose).toHaveBeenCalledOnce()
    expect(returnFocus).toHaveBeenCalledOnce()
  })

  it("leaves Escape to a dialog opened from it", async () => {
    mockApi()
    layout(900, 700)
    const { onClose } = renderDrawer()
    const list = await screen.findByRole("listbox", { name: "LoRA" }, slow)
    await within(list).findAllByRole("option", {}, slow)
    act(() => list.focus())
    fireEvent.keyDown(list, { key: "i" })
    const dialog = await screen.findByRole("dialog", { name: /watercolor|ink|neon|pastel|cyber|retro|gothic|sketch|glow|dream|storm|velvet|comic|film|mecha|floral/ }, slow)
    fireEvent.keyDown(dialog, { key: "Escape" })
    await waitFor(() => expect(screen.queryByRole("dialog", { name: dialog.getAttribute("aria-label") ?? "" })).toBeNull(), slow)
    expect(onClose).not.toHaveBeenCalled()
  })

  it("resizes from the keyboard", async () => {
    desktop()
    mockApi()
    layout(900, 700)
    renderDrawer()
    const handle = screen.getByRole("separator", { name: "Resize the networks panel" })
    fireEvent.keyDown(handle, { key: "ArrowLeft" })
    expect(useNetworksUi.getState().drawerWidth).toBe(DRAWER_WIDTH.default + 16)
    fireEvent.keyDown(handle, { key: "ArrowRight", shiftKey: true })
    expect(useNetworksUi.getState().drawerWidth).toBe(DRAWER_WIDTH.default + 16 - 64)
    fireEvent.doubleClick(handle)
    expect(useNetworksUi.getState().drawerWidth).toBe(DRAWER_WIDTH.default)
  })

  it("toggles with Alt+N", async () => {
    mockApi()
    layout(900, 700)
    render(
      <QueryClientProvider client={queryClient}>
        <NetworksDrawerHost tab="txt2img" />
      </QueryClientProvider>,
    )
    fireEvent.keyDown(window, { key: "n", code: "KeyN", altKey: true })
    expect(useNetworksUi.getState().drawerOpen).toBe(true)
    await screen.findByRole("dialog", { name: "Networks" }, slow)
    fireEvent.keyDown(window, { key: "n", code: "KeyN", altKey: true })
    expect(useNetworksUi.getState().drawerOpen).toBe(false)
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Networks" })).toBeNull())
    // other modifiers are someone else's shortcut
    fireEvent.keyDown(window, { key: "n", code: "KeyN", altKey: true, ctrlKey: true })
    expect(useNetworksUi.getState().drawerOpen).toBe(false)
    // not over another dialog (e.g. the image viewer): it would open beneath it and take the focus
    const viewer = document.createElement("div")
    viewer.setAttribute("role", "dialog")
    document.body.append(viewer)
    fireEvent.keyDown(window, { key: "n", code: "KeyN", altKey: true })
    expect(useNetworksUi.getState().drawerOpen).toBe(false)
    viewer.remove()
  })
})
