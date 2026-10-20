/**
 * Schema-declared image layers through the renderer (schema-format.md §1.3):
 * an image with `sketch` / `mask` inputs naming it as `source` renders one
 * image editor with those layers, the layer inputs render nothing, a mask
 * without `source` stays an upload field, and the editor's writes land on
 * the right params.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { beforeAll, describe, expect, it, vi } from "vitest"
import { createFormStore, type FormStore } from "@/ui/form-store"
import { TabForm } from "@/ui/SchemaForm"
import { preloadViews } from "@/ui/lazy-views"
import type { Node, TabSchema } from "@/ui/schema"
import type { ImageEditorProps } from "../types"

// these forms show deferred node views (image, file, color, JSON, tabs, outputs): fetched up front,
// they render at once as the assertions expect (the deferred path: ui/lazy-views.test.tsx)
beforeAll(() => preloadViews())

const editors = vi.hoisted(() => ({ last: null as ImageEditorProps | null, count: 0 }))

// The real editor is a lazy Konva chunk: a stub shows what the renderer hands it.
vi.mock("../ImageEditor", () => ({
  ImageEditor: (props: ImageEditorProps) => {
    editors.last = props
    editors.count++
    return <div role="group" aria-label={props.label} data-doc={props.docKey} data-active={props.activeLayer ?? ""} />
  },
}))

const layout: Node = {
  type: "column",
  children: [
    { type: "toggle", name: "colors", label: "Colors" },
    { type: "image", name: "picture", label: "Picture", kind: "image" },
    { type: "image", name: "paint", label: "Paint mask", kind: "mask", source: "picture" },
    { type: "image", name: "tint", label: "Tint", kind: "sketch", source: "picture", visible_if: { param: "colors", eq: true } },
    { type: "image", name: "external", label: "External mask", kind: "mask" },
  ],
}

function renderScript(): FormStore {
  const schema: TabSchema = {
    version: 1,
    tab: "img2img",
    core: { owner: "core", layout: { type: "column", children: [] }, defaults: {} },
    scripts: [{ id: "layers_script", title: "Layers script", kind: "alwayson", layout, defaults: { colors: false } }],
  }
  const store = createFormStore(`schema-editor-${Math.random()}`, { persistDrafts: false })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, enabled: false } } })
  render(
    <QueryClientProvider client={client}>
      <TabForm schema={schema} formId="img2img" store={store} />
    </QueryClientProvider>,
  )
  return store
}

const values = (store: FormStore) => store.getState().values.layers_script ?? {}

describe("schema image layers", () => {
  it("paints the layers over their source in one editor; layers render nothing; a sourceless mask is an upload", () => {
    renderScript()
    const sections = screen.queryAllByRole("button", { name: /Layers script/ })
    for (const section of sections) if (section.getAttribute("aria-expanded") === "false") fireEvent.click(section)
    const editor = screen.getByRole("group", { name: "Picture" })
    expect(editor.dataset.doc).toBe("img2img:layers_script:picture")
    expect(editor.dataset.active).toBe("paint")
    expect(editors.last?.pendingKey).toBe("img2img")
    expect(Object.keys(editors.last?.layers ?? {})).toEqual(["paint", "tint"])
    expect(editors.last?.layers?.paint).toMatchObject({ kind: "mask", value: null, label: "Paint mask" })
    // One editor: no upload fields for the painted layers, one for the sourceless mask.
    expect(screen.queryByText("Paint mask")).toBeNull()
    expect(screen.queryByText("Tint")).toBeNull()
    expect(screen.getByText("External mask")).toBeInTheDocument()
    expect(screen.getAllByText("Drop an image, paste, or click to browse")).toHaveLength(1)
  })

  it("offers the visible layers to pick from and writes the editor's values to their params", async () => {
    const store = renderScript()
    const sections = screen.queryAllByRole("button", { name: /Layers script/ })
    for (const section of sections) if (section.getAttribute("aria-expanded") === "false") fireEvent.click(section)
    // One visible layer: no picker.
    expect(screen.queryByRole("radiogroup", { name: "Layer to edit" })).toBeNull()
    act(() => store.getState().setValues("layers_script", { colors: true }))
    const picker = screen.getByRole("radiogroup", { name: "Layer to edit" })
    fireEvent.click(screen.getByRole("radio", { name: "Tint" }))
    expect(picker).toBeInTheDocument()
    expect(screen.getByRole("group", { name: "Picture" }).dataset.active).toBe("tint")

    act(() => editors.last?.onLayerChange?.("tint", "tint-upload"))
    expect(values(store).tint).toBe("tint-upload")
    // A new picture clears the layers painted over the old one.
    act(() => store.getState().setValues("layers_script", { picture: "old", paint: "painted" }))
    await act(async () => editors.last?.onImageChange?.({ id: "new", width: 64, height: 64 }))
    await vi.waitFor(() => expect(values(store)).toMatchObject({ picture: "new", paint: null, tint: null }))
  })
})
