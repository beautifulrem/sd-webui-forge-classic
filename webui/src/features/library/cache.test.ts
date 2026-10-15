import { InfiniteQueryObserver } from "@tanstack/react-query"
import { afterEach, describe, expect, it, vi } from "vitest"
import { queryClient } from "@/api/queries"
import type { GalleryItem, GalleryQuery } from "@/api/types"
import { batches, galleryKeys, MAX_BATCH, onNewOutputs, onOutputsChanged } from "./cache"
import { downloadZip, galleryListQuery, removeOutputs, starOutputs } from "./queries"

vi.mock("@/features/gallery/files", () => ({ downloadFile: vi.fn(), copyText: vi.fn() }))

type Call = { method: string; path: string; query: URLSearchParams; body: unknown }

const item = (id: string, extra: Partial<GalleryItem> = {}): GalleryItem => ({
  id,
  name: `${id}.png`,
  width: 64,
  height: 64,
  size: 1,
  mime: "image/png",
  created_at: "2026-10-15T08:00:00Z",
  category: "txt2img",
  model: null,
  starred: false,
  infotext: null,
  ...extra,
})

function mockFetch(answer: (call: Call) => unknown = () => ({ count: 1 })) {
  const calls: Call[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const request = input as Request
    const url = new URL(request.url)
    const text = request.method === "GET" ? "" : await request.text()
    const call = { method: request.method, path: url.pathname.replace("/api/v2", ""), query: url.searchParams, body: text ? JSON.parse(text) : undefined }
    calls.push(call)
    const data = answer(call)
    if (data instanceof Response) return data
    return new Response(JSON.stringify(data ?? {}), { status: 200, headers: { "Content-Type": "application/json" } })
  })
  return calls
}

function seed(query: GalleryQuery, pages: GalleryItem[][], total: number) {
  queryClient.setQueryData(galleryKeys.list(query), {
    pages: pages.map((items, index) => ({ items, next_cursor: index < pages.length - 1 ? `c${index}` : null, total: index === 0 ? total : null })),
    pageParams: pages.map((_, index) => (index === 0 ? null : `c${index - 1}`)),
  })
}

const cached = (query: GalleryQuery) =>
  queryClient.getQueryData<{ pages: { items: GalleryItem[]; total?: number | null }[] }>(galleryKeys.list(query))!

const ids = (count: number, prefix = "i") => Array.from({ length: count }, (_, index) => `${prefix}${index}`)

afterEach(() => {
  vi.restoreAllMocks()
  queryClient.clear()
})

describe("gallery batches", () => {
  it("splits ids into requests the server takes", () => {
    expect(batches(ids(4500)).map((batch) => batch.length)).toEqual([2000, 2000, 500])
    expect(batches([])).toEqual([])
  })

  it("stars more than 2000 images in batches, patching the pages without refetching them", async () => {
    const query: GalleryQuery = { sort: "newest" }
    seed(query, [[item("i0"), item("i1")], [item("i2")]], 3)
    const calls = mockFetch()
    await starOutputs(ids(4500), true)
    const stars = calls.filter((call) => call.path === "/gallery/star")
    expect(stars.map((call) => (call.body as { ids: string[] }).ids.length)).toEqual([2000, 2000, 500])
    expect(cached(query).pages.flatMap((page) => page.items.map((entry) => entry.starred))).toEqual([true, true, true])
    expect(calls.some((call) => call.path === "/gallery")).toBe(false)

    // The server's echo of this change needs no refresh either.
    calls.length = 0
    await onOutputsChanged(ids(200).slice(0, 199))
    expect(calls.filter((call) => call.path.startsWith("/gallery/i"))).toEqual([])
  })

  it("an unstarred image leaves a starred-only list", async () => {
    const query: GalleryQuery = { sort: "newest", starred: true }
    seed(query, [[item("a", { starred: true }), item("b", { starred: true })]], 2)
    mockFetch()
    await starOutputs(["a"], false)
    expect(cached(query).pages[0]!.items.map((entry) => entry.id)).toEqual(["b"])
    expect(cached(query).pages[0]!.total).toBe(1)
  })

  it("removes more than 2000 images in batches and drops them from the pages", async () => {
    const query: GalleryQuery = { sort: "newest" }
    seed(query, [[item("i0"), item("keep")], [item("i2001")]], 3)
    const calls = mockFetch((call) => ({ count: (call.body as { ids?: string[] } | undefined)?.ids?.length ?? 0 }))
    expect(await removeOutputs(ids(2500), true)).toBe(true)
    expect(calls.filter((call) => call.path === "/gallery/delete").map((call) => (call.body as { ids: string[] }).ids.length)).toEqual([2000, 500])
    expect(cached(query).pages.flatMap((page) => page.items.map((entry) => entry.id))).toEqual(["keep"])
    expect(cached(query).pages[0]!.total).toBe(1)
  })

  it("packs at most 2000 images into a zip", async () => {
    const calls = mockFetch(() => ({ id: "zip", name: "gallery.zip", count: 2000 }))
    await downloadZip(ids(2500))
    const zips = calls.filter((call) => call.path === "/gallery/zip")
    expect(zips).toHaveLength(1)
    expect((zips[0]!.body as { ids: string[] }).ids).toHaveLength(MAX_BATCH)
  })
})

describe("gallery cache sync", () => {
  it("merges new outputs into a newest-first list with one request", async () => {
    const query: GalleryQuery = { sort: "newest" }
    const observer = new InfiniteQueryObserver(queryClient, galleryListQuery(query))
    const unsubscribe = observer.subscribe(() => {})
    seed(query, [[item("old1"), item("old2")], [item("old3")]], 3)
    const calls = mockFetch((call) => (call.path === "/gallery" ? { items: [item("new1"), item("new2"), item("old1")], next_cursor: "x", total: 5 } : {}))
    await onNewOutputs()
    const lists = calls.filter((call) => call.path === "/gallery")
    expect(lists).toHaveLength(1)
    expect(lists[0]!.query.get("cursor")).toBeNull()
    const data = cached(query)
    expect(data.pages.map((page) => page.items.map((entry) => entry.id))).toEqual([["new1", "new2", "old1", "old2"], ["old3"]])
    expect(data.pages[0]!.total).toBe(5)
    unsubscribe()
  })

  it("drops lists that are not on screen (they start over at their first page)", async () => {
    const query: GalleryQuery = { sort: "newest" }
    seed(query, [[item("a")]], 1)
    const calls = mockFetch()
    await onNewOutputs()
    expect(queryClient.getQueryData(galleryKeys.list(query))).toBeUndefined()
    expect(calls.filter((call) => call.path === "/gallery")).toEqual([])
  })

  it("patches an image changed by another client from the server", async () => {
    const query: GalleryQuery = { sort: "newest" }
    seed(query, [[item("a"), item("b")]], 2)
    const calls = mockFetch((call) =>
      call.path === "/gallery/a" ? item("a", { starred: true }) : call.path === "/gallery/b" ? new Response(JSON.stringify({ detail: "gone" }), { status: 404, headers: { "Content-Type": "application/json" } }) : {},
    )
    await onOutputsChanged(["a", "b", "not-loaded"])
    const fetched = calls.map((call) => call.path)
    expect(fetched).toEqual(expect.arrayContaining(["/gallery/a", "/gallery/b"]))
    expect(fetched).not.toContain("/gallery/not-loaded") // not shown anywhere: nothing to patch
    expect(cached(query).pages[0]!.items).toEqual([item("a", { starred: true })])
  })
})
