/**
 * The Generate page's layouts and wiring: phone / tablet / desktop panes, the mode switch in the
 * header, the pane toggles (with the run controls moving onto the stage), and the image viewer
 * (lightbox, or a new tab when the lightbox is off).
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, cleanup, configure, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { useProgress } from "@/api/progress"
import { keys, queryClient } from "@/api/queries"
import type { Bootstrap } from "@/api/types"
import { useHeaderSlotTarget } from "@/app/header-slot-store"
import { useSession } from "@/features/gallery/session-store"
import { resetRunControllers } from "@/features/progress/run-controller"
import { bootstrapWith } from "@/test/api-fixtures"
import { recordFetch } from "@/test/fetch-recorder"
import GeneratePage from "./index"

configure({ asyncUtilTimeout: 10_000 })

const observers = new Set<{ report: () => void }>()

/** Tells the ResizeObservers about the current sizes, as a browser does after a layout change. */
const relayout = () => act(() => observers.forEach((observer) => observer.report()))

/**
 * jsdom has no layout: the pane group is `width` px wide and each pane takes its flex share of it;
 * ResizeObservers report the sizes on `observe` and on `relayout()`.
 */
function paneLayout(width: number) {
  vi.stubGlobal(
    "ResizeObserver",
    class {
      /** Each observed element and the width last reported for it (a browser reports changes only). */
      private readonly targets = new Map<HTMLElement, number | null>()
      private readonly callback: ResizeObserverCallback
      constructor(callback: ResizeObserverCallback) {
        this.callback = callback
        observers.add(this)
      }
      report() {
        const entries: ResizeObserverEntry[] = []
        for (const [target, last] of this.targets) {
          const size = { inlineSize: target.offsetWidth, blockSize: 800 }
          if (size.inlineSize === last) continue
          this.targets.set(target, size.inlineSize)
          entries.push({ target, borderBoxSize: [size], contentBoxSize: [size], contentRect: { width: size.inlineSize, height: 800 } } as unknown as ResizeObserverEntry)
        }
        if (entries.length > 0) this.callback(entries, this as unknown as ResizeObserver)
      }
      observe(target: HTMLElement) {
        this.targets.set(target, null)
        setTimeout(() => act(() => this.report()))
      }
      unobserve(target: HTMLElement) {
        this.targets.delete(target)
      }
      disconnect() {
        this.targets.clear()
        observers.delete(this)
      }
    },
  )
  vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockImplementation(function (this: HTMLElement) {
    if (this.dataset.slot === "resizable-panel-group") return width
    if (this.dataset.slot === "resizable-panel") return (width * Number.parseFloat(this.style.flexGrow || "0")) / 100
    return 0
  })
}

/** The pane resize handles (forms have separators of their own). */
const paneHandles = () => [...document.querySelectorAll<HTMLElement>('[data-slot="resizable-handle"]')].map((handle) => handle.getAttribute("aria-label"))
/** The stage's image and its expand button both open the viewer. */
const openViewerButton = async () => (await screen.findAllByRole("button", { name: "Open image viewer" }))[0]!

/** The viewport: `matchMedia` answers the page's width queries for this width (rem = 16 px). */
function viewport(width: number) {
  vi.spyOn(window, "matchMedia").mockImplementation((query: string) => {
    const min = /min-width:\s*([\d.]+)rem/.exec(query)
    const matches = min ? width >= Number(min[1]) * 16 : false
    return { matches, media: query, onchange: null, addEventListener: () => {}, removeEventListener: () => {}, addListener: () => {}, removeListener: () => {}, dispatchEvent: () => false } as MediaQueryList
  })
}

function renderPage(tab: "txt2img" | "img2img" = "txt2img", options: Record<string, unknown> = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } })
  for (const form of ["txt2img", "img2img"] as const) {
    client.setQueryData(keys.schema(form), tabSchema(form))
    queryClient.setQueryData(keys.schema(form), tabSchema(form))
  }
  client.setQueryData<Bootstrap>(keys.app, bootstrapWith(options))
  queryClient.setQueryData<Bootstrap>(keys.app, bootstrapWith(options))
  const header = document.createElement("div")
  header.dataset.testid = "header"
  document.body.append(header)
  act(() => useHeaderSlotTarget.getState().setElement(header))
  const view = render(
    <QueryClientProvider client={client}>
      <GeneratePage route={{ name: "generate", tab }} />
    </QueryClientProvider>,
  )
  return { ...view, header }
}

/** Two finished images in the session history (the newest is selected). */
function finishJob() {
  act(() => {
    useProgress.getState().track("job-1", "txt2img")
    useProgress.getState().completed({ job_id: "job-1", outputs: [{ id: "out-a", kind: "image" }, { id: "out-b", kind: "image" }], infotext: "a cat\nSteps: 20, Seed: 42, Size: 512x512" })
  })
}

beforeEach(() => {
  recordFetch(({ path }) => (path === "/api/v2/jobs" ? { body: { items: [] } } : undefined))
  useProgress.setState({ running: null, owned: {}, last: {}, orphans: {} })
  useSession.getState().clear()
  resetRunControllers()
})
afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  observers.clear()
  cleanup()
  act(() => useHeaderSlotTarget.getState().setElement(null))
  for (const header of document.querySelectorAll("[data-testid=header]")) header.remove()
  queryClient.clear()
})

describe("generate page layout", { timeout: 30_000 }, () => {
  it("puts the txt2img / img2img switch in the header, marking the current mode", () => {
    viewport(400)
    const { header } = renderPage("img2img")
    const nav = within(header).getByRole("navigation", { name: "Mode" })
    const links = within(nav).getAllByRole("link")
    expect(links.map((link) => [link.textContent, link.getAttribute("aria-current")])).toEqual([
      ["txt2img", null],
      ["img2img", "page"],
    ])
    expect(links[0]).toHaveAttribute("href", expect.stringContaining("txt2img"))
  })

  it("phones: one column (stage, history strip, parameters) with the run controls at the bottom", async () => {
    viewport(400)
    renderPage()
    expect(paneHandles()).toEqual([])
    expect(screen.queryByRole("button", { name: "Hide parameters" })).toBeNull()
    const stage = screen.getByRole("region", { name: "Result" })
    const history = screen.getByRole("region", { name: "History" })
    const generate = await screen.findAllByRole("button", { name: /^Generate/ })
    // document order: stage, history, then the parameters with the sticky bar after them
    expect(stage.compareDocumentPosition(history) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(history.compareDocumentPosition(generate[0]!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(generate[0]!.closest(".sticky")).toHaveClass("bottom-0")
  })

  it("tablets: parameters | stage with the history as a strip under it", async () => {
    viewport(1000)
    renderPage()
    await waitFor(() => expect(paneHandles()).toEqual(["Resize the parameters panel"]))
    expect(screen.getByRole("button", { name: "Hide parameters" })).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Hide history" })).toBeNull()
    expect(screen.getByRole("region", { name: "History" })).toBeInTheDocument()
    const generate = await screen.findAllByRole("button", { name: /^Generate/ })
    expect(generate[0]!.closest(".sticky")).toHaveClass("top-0")
  })

  it("desktops: three panes; hiding the parameters moves the run controls onto the stage", async () => {
    viewport(1400)
    paneLayout(1400)
    renderPage()
    await waitFor(() => expect(paneHandles()).toEqual(["Resize the parameters panel", "Resize the history panel"]))
    const generateButtons = () => screen.queryAllByRole("button", { name: /^Generate/ }).filter((button) => !button.hasAttribute("aria-label"))
    const onStage = () => generateButtons().filter((button) => screen.getByRole("region", { name: "Result" }).contains(button))
    await waitFor(() => expect(generateButtons()).toHaveLength(1))
    expect(onStage()).toEqual([])

    fireEvent.click(screen.getByRole("button", { name: "Hide history" }))
    relayout()
    fireEvent.click(await screen.findByRole("button", { name: "Show history" }))
    relayout()
    expect(await screen.findByRole("button", { name: "Hide history" })).toHaveAttribute("aria-pressed", "true")

    fireEvent.click(screen.getByRole("button", { name: "Hide parameters" }))
    relayout()
    const show = await screen.findByRole("button", { name: "Show parameters" })
    expect(show).toHaveAttribute("aria-pressed", "false")
    // the run controls are now in the stage's toolbar
    await waitFor(() => expect(onStage()).toHaveLength(1))
    fireEvent.click(show)
    relayout()
    expect(await screen.findByRole("button", { name: "Hide parameters" })).toHaveAttribute("aria-pressed", "true")
    await waitFor(() => expect(onStage()).toEqual([]))
  })
})

describe("image viewer", { timeout: 30_000 }, () => {
  it("opens the lightbox on the selected result, or on the one opened from the history", async () => {
    viewport(1400)
    renderPage()
    finishJob()
    fireEvent.click(await openViewerButton())
    expect(await screen.findByRole("dialog", { name: "Image viewer" })).toBeInTheDocument()
    expect(useSession.getState().follow).toBe(true)

    fireEvent.click(screen.getByRole("button", { name: "Close" }))
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Image viewer" })).toBeNull())
    // the history opens the result picked there (Enter on the list, as a double click)
    act(() => useSession.getState().select("job-1:1"))
    fireEvent.keyDown(screen.getByRole("listbox", { name: "History" }), { key: "Enter" })
    expect(await screen.findByRole("dialog", { name: "Image viewer" })).toBeInTheDocument()
    // opening a result stops following new ones
    expect(useSession.getState()).toMatchObject({ selected: "job-1:1", follow: false })
  })

  it("opens the image in a new tab when the lightbox is off (js_modal_lightbox)", async () => {
    viewport(1400)
    const open = vi.spyOn(window, "open").mockReturnValue(null)
    renderPage("txt2img", { js_modal_lightbox: false })
    finishJob()
    fireEvent.click(await openViewerButton())
    expect(open).toHaveBeenCalledWith("/api/v2/files/out-a", "_blank", "noopener")
    expect(screen.queryByRole("dialog", { name: "Image viewer" })).toBeNull()
  })
})
