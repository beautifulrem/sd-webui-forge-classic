import { QueryClientProvider } from "@tanstack/react-query"
import { act, configure, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { queryClient } from "@/api/queries"
import type { GalleryItem } from "@/api/types"
import { useGalleryView } from "@/features/library/filters"
import GalleryPage from "./index"

type Call = { method: string; path: string; query: URLSearchParams; body: unknown }

// Layout in jsdom is slow when the whole suite runs in parallel.
configure({ asyncUtilTimeout: 5_000 })

const item = (index: number, extra: Partial<GalleryItem> = {}): GalleryItem => ({
  id: `img${index}`,
  name: `0000${index}.png`,
  width: index % 2 ? 832 : 1216,
  height: index % 2 ? 1216 : 832,
  size: 1000,
  mime: "image/png",
  created_at: "2026-10-15T08:00:00Z",
  category: "txt2img",
  model: "anima",
  starred: false,
  infotext: `prompt number ${index}\nSteps: 20, Seed: ${index}, Model: anima`,
  ...extra,
})

function mockServer(items: GalleryItem[]) {
  const calls: Call[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const request = input as Request
    const url = new URL(request.url)
    const path = url.pathname.replace("/api/v2", "")
    const text = request.method === "GET" ? "" : await request.text()
    calls.push({ method: request.method, path, query: url.searchParams, body: text ? JSON.parse(text) : undefined })
    let data: unknown = {}
    if (path === "/gallery") data = { items, next_cursor: null, total: items.length }
    else if (path === "/gallery/facets")
      data = { models: [{ name: "anima", count: items.length }], categories: [{ name: "txt2img", count: items.length }], oldest: null, newest: null, total: items.length, starred: 0 }
    else if (path === "/gallery/star" || path === "/gallery/delete") data = { count: 1 }
    return new Response(JSON.stringify(data), { status: 200, headers: { "Content-Type": "application/json" } })
  })
  return calls
}

function renderPage() {
  render(
    <QueryClientProvider client={queryClient}>
      <GalleryPage />
    </QueryClientProvider>,
  )
}

describe("gallery page", { timeout: 20_000 }, () => {
  beforeEach(() => {
    useGalleryView.setState({ q: "", category: null, model: null, from: null, to: null, starred: false, sort: "newest", layout: "justified", size: "m" })
    // jsdom has no layout: give the grid a size so the virtualizer renders rows.
    vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockReturnValue(900)
    vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockReturnValue(900)
    vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockReturnValue(700)
  })
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("lays images out in justified rows at their aspect ratio", async () => {
    mockServer([item(1), item(2), item(3), item(4)])
    renderPage()
    const tiles = await screen.findAllByRole("listitem")
    expect(tiles).toHaveLength(4)
    const widths = tiles.map((tile) => Number.parseFloat(tile.style.width))
    const heights = tiles.map((tile) => Number.parseFloat(tile.style.height))
    // A portrait tile is narrower than a landscape one in the same row.
    expect(heights[0]).toBeCloseTo(heights[1]!)
    expect(widths[0]! / heights[0]!).toBeCloseTo(832 / 1216, 2)
    expect(screen.getByRole("button", { name: "prompt number 1" })).toBeInTheDocument()
    expect(screen.getByText("Images: 4")).toBeInTheDocument()
  })

  it("selects with the check boxes and acts on the selection", async () => {
    const calls = mockServer([item(1), item(2), item(3)])
    renderPage()
    fireEvent.click(await screen.findByRole("checkbox", { name: "Select 00001.png" }))
    fireEvent.click(screen.getByRole("checkbox", { name: "Select 00003.png", hidden: true }))
    const bar = screen.getByRole("toolbar", { name: "Selected images" })
    expect(bar).toHaveTextContent("2 selected")
    expect(within(bar).getByRole("button", { name: "Compare" })).toBeInTheDocument()

    fireEvent.click(within(bar).getByRole("button", { name: "Star" }))
    await waitFor(() => expect(calls.find((call) => call.path === "/gallery/star")?.body).toEqual({ ids: ["img1", "img3"], starred: true }))

    fireEvent.click(within(bar).getByRole("button", { name: "Remove…" }))
    const dialog = await screen.findByRole("alertdialog")
    expect(dialog).toHaveTextContent("Remove 2 images?")
    fireEvent.click(within(dialog).getByRole("checkbox"))
    fireEvent.click(within(dialog).getByRole("button", { name: "Delete files" }))
    await waitFor(() => expect(calls.find((call) => call.path === "/gallery/delete")?.body).toEqual({ ids: ["img1", "img3"], delete_files: true }))
  })

  it("a click with a selection extends it; Escape clears it", async () => {
    mockServer([item(1), item(2), item(3)])
    renderPage()
    fireEvent.click(await screen.findByRole("checkbox", { name: "Select 00001.png" }))
    fireEvent.click(screen.getByRole("button", { name: "prompt number 3" }), { shiftKey: true })
    expect(screen.getByRole("toolbar", { name: "Selected images" })).toHaveTextContent("3 selected")
    fireEvent.keyDown(window, { key: "Escape" })
    expect(screen.queryByRole("toolbar", { name: "Selected images" })).not.toBeInTheDocument()
  })

  it("is one tab stop moved with the arrow keys; Space selects, F stars", async () => {
    const calls = mockServer([item(1), item(2), item(3), item(4), item(5), item(6)])
    renderPage()
    const first = await screen.findByRole("button", { name: "prompt number 1" })
    const tiles = screen.getAllByRole("button", { name: /^prompt number/ })
    expect(tiles.filter((tile) => tile.tabIndex === 0)).toEqual([first])
    expect(screen.getByRole("checkbox", { name: "Select 00001.png", hidden: true }).tabIndex).toBe(-1)

    first.focus()
    fireEvent.keyDown(first, { key: "ArrowRight" })
    const second = screen.getByRole("button", { name: "prompt number 2" })
    expect(second).toHaveFocus()
    expect(second.tabIndex).toBe(0)
    fireEvent.keyDown(second, { key: "ArrowDown" })
    expect(document.activeElement).not.toBe(second) // the image below, in the next row
    fireEvent.keyDown(document.activeElement!, { key: "Home" })
    expect(first).toHaveFocus()

    fireEvent.keyDown(first, { key: " " })
    expect(screen.getByRole("toolbar", { name: "Selected images" })).toHaveTextContent("1 selected")
    fireEvent.keyDown(first, { key: "f" })
    await waitFor(() => expect(calls.find((call) => call.path === "/gallery/star")?.body).toEqual({ ids: ["img1"], starred: true }))
  })

  it("keeps a tab stop on a tile in view after scrolling, and keeps a focused tile rendered", async () => {
    mockServer(Array.from({ length: 160 }, (_, index) => item(index + 1)))
    renderPage()
    const first = await screen.findByRole("button", { name: "prompt number 1" })
    const scroller = screen.getByTestId("gallery-scroller")
    const tabStops = () => screen.getAllByRole("button", { name: /^prompt number/ }).filter((tile) => tile.tabIndex === 0)

    // Scrolled far away with the mouse: the first tile is gone, a tile in view takes the tab stop.
    scroller.scrollTop = 4000
    fireEvent.scroll(scroller)
    await waitFor(() => expect(screen.queryByRole("button", { name: "prompt number 1" })).not.toBeInTheDocument())
    expect(tabStops()).toHaveLength(1)

    // A focused tile stays rendered (and focused) when it scrolls out of view.
    scroller.scrollTop = 0
    fireEvent.scroll(scroller)
    const again = await screen.findByRole("button", { name: "prompt number 1" })
    again.focus()
    scroller.scrollTop = 4000
    fireEvent.scroll(scroller)
    await waitFor(() => expect(tabStops().length).toBeGreaterThan(0))
    expect(screen.getByRole("button", { name: "prompt number 1" })).toHaveFocus()
    expect(first.isConnected || again.isConnected).toBe(true)
  })

  it("starts a new grid with the first image as the tab stop when the filters change", async () => {
    mockServer([item(1), item(2), item(3)])
    renderPage()
    const second = await screen.findByRole("button", { name: "prompt number 2" })
    second.focus()
    fireEvent.keyDown(second, { key: "End" })
    expect(screen.getByRole("button", { name: "prompt number 3" }).tabIndex).toBe(0)
    act(() => useGalleryView.setState({ sort: "oldest" }))
    await waitFor(() => expect(screen.getByRole("button", { name: "prompt number 1" }).tabIndex).toBe(0))
    expect(screen.getAllByRole("button", { name: /^prompt number/ }).filter((tile) => tile.tabIndex === 0)).toHaveLength(1)
  })

  it("sends the filters to the server", async () => {
    const calls = mockServer([item(1)])
    useGalleryView.setState({ starred: true, model: "anima", sort: "oldest", from: "2026-10-01" })
    renderPage()
    await screen.findAllByRole("listitem")
    const query = calls.find((call) => call.path === "/gallery")!.query
    expect(query.get("starred")).toBe("true")
    expect(query.get("model")).toBe("anima")
    expect(query.get("sort")).toBe("oldest")
    expect(query.get("since")).toMatch(/^2026-09-30T|^2026-10-01T/)
    expect(screen.getByRole("button", { name: "Clear filters" })).toBeInTheDocument()
  })

  it("explains an empty library", async () => {
    mockServer([])
    renderPage()
    expect(await screen.findByText("No images yet")).toBeInTheDocument()
  })
})
