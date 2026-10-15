import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { createFormStore, type FormStore } from "@/ui/form-store"
import { TabForm } from "@/ui/SchemaForm"
import type { TabSchema } from "@/ui/schema"
import { widgetKeys } from "../_lib/i18n"
import zh from "./locales/zh-CN.json"
import source from "./index.tsx?raw"
import { moveItem, removeAt, replaceAt, selectionAfterRemove } from "./list"

// Widgets load lazily; give the chunk time on a busy machine
const LOAD = { timeout: 15_000 }

describe("list edits", () => {
  it("moves, removes and replaces", () => {
    expect(moveItem(["a", "b", "c"], 0, 2)).toEqual(["b", "c", "a"])
    expect(moveItem(["a", "b", "c"], 2, 0)).toEqual(["c", "a", "b"])
    expect(moveItem(["a", "b", "c"], 1, 9)).toEqual(["a", "c", "b"])
    expect(moveItem(["a"], 3, 0)).toEqual(["a"])
    expect(removeAt(["a", "b", "c"], 1)).toEqual(["a", "c"])
    expect(replaceAt(["a", "b"], 1, "x")).toEqual(["a", "x"])
    expect(selectionAfterRemove(2, 3)).toBe(1)
    expect(selectionAfterRemove(0, 3)).toBe(0)
    expect(selectionAfterRemove(0, 1)).toBe(-1)
  })
})

const schema: TabSchema = {
  version: 1,
  tab: "img2img",
  core: { owner: "core", layout: { type: "column", children: [{ type: "slot", section: "scripts" }] }, defaults: {} },
  scripts: [
    {
      id: "imagestitch_integrated",
      title: "ImageStitch Integrated",
      kind: "alwayson",
      section: null,
      order: 0,
      layout: {
        type: "group",
        title: "ImageStitch Integrated",
        open: true,
        children: [{ type: "image_list", name: "references", label: "Reference Image(s)", default: [], widget: "image-list", props: { columns: 3 } }],
      },
      defaults: { references: [] },
      args: ["references"],
    },
  ],
}

let uploaded = 0
const uploadResponse = (id: string) => new Response(JSON.stringify({ id }), { status: 201, headers: { "Content-Type": "application/json" } })

function mockUploads() {
  vi.spyOn(globalThis, "fetch").mockImplementation(async () => uploadResponse(`up-${++uploaded}`))
}

/** Uploads that answer only when the test releases them, in any order. */
function heldUploads() {
  const held: (() => void)[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(() => {
    const id = `held-${held.length + 1}`
    return new Promise<Response>((resolve) => held.push(() => resolve(uploadResponse(id))))
  })
  return held
}

const image = (name = "a.png") => new File([new Uint8Array([1])], name, { type: "image/png" })

function renderForm(references: string[] = []): FormStore {
  const store = createFormStore(`image-list-${Math.random()}`, { persistDrafts: false })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, enabled: false } } })
  render(
    <QueryClientProvider client={client}>
      <TabForm schema={{ ...schema, scripts: [{ ...schema.scripts[0]!, defaults: { references } }] }} formId="img2img" store={store} />
    </QueryClientProvider>,
  )
  return store
}

const refs = (store: FormStore) => store.getState().values.imagestitch_integrated?.references as string[] | undefined

describe("image-list widget", { timeout: 30_000 }, () => {
  afterEach(() => vi.restoreAllMocks())

  it("appends uploads in order", async () => {
    mockUploads()
    const store = renderForm()
    const zone = await screen.findByRole("button", { name: "Add images" }, LOAD)
    const input = zone.querySelector("input[type=file]") as HTMLInputElement
    fireEvent.change(input, { target: { files: [image("1.png"), image("2.png")] } })
    await waitFor(() => expect(refs(store)).toHaveLength(2))
    const [first, second] = refs(store) ?? []
    expect(first).toMatch(/^up-/)
    expect(screen.getAllByRole("option").map((option) => option.getAttribute("aria-label"))).toEqual(["Image 1", "Image 2"])
    // The last added image is selected.
    expect(screen.getByRole("option", { name: "Image 2" })).toHaveAttribute("aria-selected", "true")
    expect(second).not.toBe(first)
  })

  it("selects, reorders with the keyboard and the toolbar, removes and clears", async () => {
    const store = renderForm(["a", "b", "c"])
    const list = await screen.findByRole("listbox", { name: "Reference Image(s)" }, LOAD)
    fireEvent.click(screen.getByRole("option", { name: "Image 1" }))
    expect(list).toHaveAttribute("aria-activedescendant", screen.getByRole("option", { name: "Image 1" }).id)

    fireEvent.keyDown(list, { key: "ArrowRight", altKey: true })
    expect(refs(store)).toEqual(["b", "a", "c"])
    expect(screen.getByRole("option", { name: "Image 2" })).toHaveAttribute("aria-selected", "true")

    fireEvent.keyDown(list, { key: "End" })
    fireEvent.click(screen.getByRole("button", { name: "Move earlier" }))
    expect(refs(store)).toEqual(["b", "c", "a"])

    fireEvent.keyDown(list, { key: "Delete" })
    expect(refs(store)).toEqual(["b", "a"])
    expect(screen.getByRole("option", { name: "Image 2" })).toHaveAttribute("aria-selected", "true")

    fireEvent.click(within(screen.getByRole("option", { name: "Image 1" })).getByTitle("Remove image 1"))
    expect(refs(store)).toEqual(["a"])

    fireEvent.click(screen.getByRole("button", { name: "Clear all" }))
    expect(refs(store)).toEqual([])
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument()
  })

  it("reorders by dragging a thumbnail onto another", async () => {
    const store = renderForm(["a", "b", "c"])
    const options = await screen.findAllByRole("option", {}, LOAD)
    const dataTransfer = { setData: () => {}, effectAllowed: "", dropEffect: "" }
    fireEvent.dragStart(options[2]!, { dataTransfer })
    fireEvent.dragOver(options[0]!, { dataTransfer })
    fireEvent.drop(options[0]!, { dataTransfer })
    expect(refs(store)).toEqual(["c", "a", "b"])
  })

  it("appends files dropped on the thumbnails", async () => {
    mockUploads()
    const store = renderForm(["a"])
    const list = await screen.findByRole("listbox", {}, LOAD)
    fireEvent.drop(list, { dataTransfer: { files: [image()], types: ["Files"] } })
    await waitFor(() => expect(refs(store)).toHaveLength(2))
    expect(refs(store)?.[0]).toBe("a")
  })

  it("replaces the selected image", async () => {
    mockUploads()
    const store = renderForm(["a", "b"])
    fireEvent.click(await screen.findByRole("option", { name: "Image 2" }, LOAD))
    await act(async () => {
      fireEvent.change(screen.getByTestId("replace-input"), { target: { files: [image()] } })
    })
    await waitFor(() => expect(refs(store)?.[1]).toMatch(/^up-/))
    expect(refs(store)?.[0]).toBe("a")
  })

  it("keeps both images of overlapping uploads", async () => {
    const held = heldUploads()
    const store = renderForm(["a"])
    const zone = await screen.findByRole("button", { name: "Add images" }, LOAD)
    const input = zone.querySelector("input[type=file]") as HTMLInputElement
    fireEvent.change(input, { target: { files: [image("1.png")] } })
    fireEvent.change(input, { target: { files: [image("2.png")] } })
    await waitFor(() => expect(held).toHaveLength(2))
    // The second upload finishes first
    await act(async () => held[1]!())
    await waitFor(() => expect(refs(store)).toEqual(["a", "held-2"]))
    await act(async () => held[0]!())
    await waitFor(() => expect(refs(store)).toEqual(["a", "held-2", "held-1"]))
  })

  it("does not bring back an image removed while its replacement uploads", async () => {
    const held = heldUploads()
    const store = renderForm(["a", "b", "c"])
    fireEvent.click(await screen.findByRole("option", { name: "Image 2" }, LOAD))
    fireEvent.change(screen.getByTestId("replace-input"), { target: { files: [image()] } })
    await waitFor(() => expect(held).toHaveLength(1))
    fireEvent.keyDown(screen.getByRole("listbox"), { key: "Delete" })
    expect(refs(store)).toEqual(["a", "c"])
    await act(async () => held[0]!())
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1))
    expect(refs(store)).toEqual(["a", "c"])
  })

  it("replaces the image it was started on after the list was reordered", async () => {
    const held = heldUploads()
    const store = renderForm(["a", "b", "c"])
    fireEvent.click(await screen.findByRole("option", { name: "Image 2" }, LOAD))
    fireEvent.change(screen.getByTestId("replace-input"), { target: { files: [image()] } })
    await waitFor(() => expect(held).toHaveLength(1))
    const list = screen.getByRole("listbox")
    fireEvent.keyDown(list, { key: "ArrowRight", altKey: true })
    expect(refs(store)).toEqual(["a", "c", "b"])
    await act(async () => held[0]!())
    await waitFor(() => expect(refs(store)).toEqual(["a", "c", "held-1"]))
  })

  it("translates every chrome string", () => {
    const missing = widgetKeys(source).filter((key) => !(key in zh))
    expect(missing).toEqual([])
  })
})
