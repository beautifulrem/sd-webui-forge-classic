import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { queryClient } from "@/api/queries"
import type { NetworkItem } from "@/api/types"
import { useSession } from "@/features/gallery/session-store"
import { PreviewDialog } from "./PreviewDialog"

type Call = { method: string; path: string; url: URL; body: unknown }

const item = (name: string): NetworkItem => ({ name, title: name, kind: "lora", path: `${name}.safetensors`, folder: "", tags: [], search_terms: [], editable: true, hidden: false })

function mockServer() {
  const calls: Call[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const request = input as Request
    const url = new URL(request.url)
    const path = url.pathname.replace("/api/v2", "")
    const body = request.headers.get("content-type")?.includes("application/json") ? await request.json() : undefined
    calls.push({ method: request.method, path, url, body })
    const json = (data: unknown) => new Response(JSON.stringify(data), { headers: { "Content-Type": "application/json" } })
    if (path === "/outputs") {
      const page = url.searchParams.get("cursor") ? 1 : 0
      const items = Array.from({ length: 3 }, (_, i) => ({ id: `out-${page}-${i}`, kind: "image", name: "x.png", created_at: "2026-10-15T00:00:00Z" }))
      return json({ items, next_cursor: page === 0 ? "page-2" : null, total: 6 })
    }
    if (path === "/uploads") return json({ id: "upload-1", kind: "image", name: "mine.png" })
    return json({ ...item("ink"), preview: "upload-1" })
  })
  return calls
}

function Harness({ item: current, open, onOpenChange = () => {} }: { item: NetworkItem | null; open: boolean; onOpenChange?: (open: boolean) => void }) {
  return (
    <QueryClientProvider client={queryClient}>
      <PreviewDialog item={current} open={open} onOpenChange={onOpenChange} />
    </QueryClientProvider>
  )
}

describe("preview dialog", { timeout: 30_000 }, () => {
  beforeEach(() => {
    queryClient.clear()
    useSession.setState({ items: [] })
  })
  afterEach(() => vi.restoreAllMocks())

  it("uploads an image, then makes it the preview", async () => {
    const calls = mockServer()
    const onOpenChange = vi.fn()
    render(<Harness item={item("ink")} open onOpenChange={onOpenChange} />)
    fireEvent.click(screen.getByRole("tab", { name: "Upload" }))
    const input = await screen.findByTestId("preview-upload")
    fireEvent.change(input, { target: { files: [new File([new Uint8Array([137, 80, 78, 71])], "mine.png", { type: "image/png" })] } })
    await waitFor(() => expect(onOpenChange).toHaveBeenCalledWith(false))
    const relevant = calls.filter((call) => call.path !== "/outputs").map((call) => `${call.method} ${call.path}`)
    expect(relevant).toEqual(["POST /uploads", "PUT /networks/lora/ink/preview"])
    expect(calls.at(-1)!.body).toEqual({ file_id: "upload-1" })
  })

  it("pages the gallery and picks with the arrow keys", async () => {
    const calls = mockServer()
    render(<Harness item={item("ink")} open />)
    const group = await screen.findByRole("radiogroup", { name: "Image" })
    expect(within(group).getAllByRole("radio")).toHaveLength(3)
    fireEvent.click(screen.getByRole("button", { name: "Load more" }))
    await waitFor(() => expect(within(group).getAllByRole("radio")).toHaveLength(6))
    expect(screen.queryByRole("button", { name: "Load more" })).toBeNull()

    const radios = within(group).getAllByRole("radio")
    expect(radios.filter((radio) => radio.tabIndex === 0)).toEqual([radios[0]])
    radios[0]!.focus()
    fireEvent.keyDown(group, { key: "ArrowRight" })
    expect(radios[0]).toHaveAttribute("aria-checked", "true")
    fireEvent.keyDown(group, { key: "ArrowRight" })
    expect(radios[1]).toHaveAttribute("aria-checked", "true")
    expect(document.activeElement).toBe(radios[1])
    fireEvent.keyDown(group, { key: "End" })
    expect(radios[5]).toHaveAttribute("aria-checked", "true")
    fireEvent.click(screen.getByRole("button", { name: "Use as preview" }))
    await waitFor(() => expect(calls.find((call) => call.method === "PUT")?.body).toEqual({ file_id: "out-1-2" }))
  })

  it("starts fresh for every item", async () => {
    mockServer()
    useSession.setState({ items: [{ key: "j:0", fileId: "result-1", jobId: "j", tab: "txt2img", index: 0, infotext: null, createdAt: 1 }] })
    const { rerender } = render(<Harness item={item("ink")} open />)
    fireEvent.click(screen.getByRole("radio", { name: "Result 1" }))
    fireEvent.click(screen.getByRole("tab", { name: "Gallery" }))
    rerender(<Harness item={item("ink")} open={false} />)
    rerender(<Harness item={item("pastel")} open />)
    // back on the session's results, nothing picked
    const result = await screen.findByRole("radio", { name: "Result 1" })
    expect(result).toHaveAttribute("aria-checked", "false")
    expect(screen.getByRole("button", { name: "Use as preview" })).toBeDisabled()
  })
})
