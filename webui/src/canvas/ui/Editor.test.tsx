/**
 * The editor's two-way binding of the document and its values, without
 * Konva (the body is a stub, rasters live in memory): the background
 * follows the value (send-to), a failed load leaves nothing editable and
 * can be retried, uploaded masks load onto the canvas, painted layers are
 * committed to their params.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import type { ComponentProps } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { getDocument } from "../engine/registry"
import type { ImageEditorProps } from "../types"
import type { EditorBody } from "./EditorBody"
import Editor from "./Editor"

type BodyProps = ComponentProps<typeof EditorBody>

const body = vi.hoisted(() => ({ props: null as null | BodyProps }))

vi.mock("./EditorBody", () => ({
  EditorBody: (props: BodyProps) => {
    body.props = props
    return (
      <div data-testid="body" data-ready={String(props.ready)} data-failed={String(props.loadFailed)} data-busy={String(props.busy)}>
        <button type="button" onClick={props.onRetry}>
          retry
        </button>
      </div>
    )
  },
}))

const W = 8
const H = 6

/**
 * Documents on memory rasters (jsdom has no 2D canvas), uploads to a counter,
 * images by upload id in the URL: "broken" ones fail to load, "mask-upload"
 * ones are opaque white (an uploaded black-and-white mask: white = masked).
 */
const loads = vi.hoisted(() => ({ calls: [] as string[] }))
vi.mock("../engine/registry", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../engine/registry")>()
  const { createMemoryRaster, testImage } = await import("../engine/memory-raster")
  const { EditorDocument } = await import("../engine/document")
  const documents = new Map<string, InstanceType<typeof EditorDocument>>()
  let uploads = 0
  return {
    ...actual,
    getDocument: (key: string) => {
      let doc = documents.get(key)
      if (!doc) {
        doc = new EditorDocument({
          createRaster: createMemoryRaster,
          sync: { encode: async (data: Uint8ClampedArray) => new Blob([data.slice()]), upload: async () => `layer-${++uploads}`, delay: 5 },
        })
        documents.set(key, doc)
      }
      return doc
    },
    loadImage: async (url: string) => {
      loads.calls.push(url)
      if (url.includes("broken")) throw new Error("decode failed")
      if (url.includes("mask-upload")) return testImage(W, H, [255, 255, 255, 255])
      return testImage(W, H, [40, 80, 120, 255])
    },
  }
})

let keySeq = 0

function setup(initial: Partial<ImageEditorProps>) {
  const docKey = `editor-test-${++keySeq}`
  const client = new QueryClient({ defaultOptions: { queries: { enabled: false } } })
  const onLayerChange = vi.fn()
  const onImageChange = vi.fn()
  const props = (over: Partial<ImageEditorProps>): ImageEditorProps => ({ docKey, image: null, onLayerChange, onImageChange, ...initial, ...over })
  const view = render(
    <QueryClientProvider client={client}>
      <Editor {...props({})} />
    </QueryClientProvider>,
  )
  const update = (over: Partial<ImageEditorProps>) =>
    view.rerender(
      <QueryClientProvider client={client}>
        <Editor {...props(over)} />
      </QueryClientProvider>,
    )
  return { doc: getDocument(docKey), update, onLayerChange, onImageChange }
}

const stub = () => screen.getByTestId("body")

afterEach(() => {
  loads.calls = []
  body.props = null
})

describe("Editor binding", () => {
  it("loads the background, then an uploaded mask onto it (luminance: white = masked)", async () => {
    const { doc } = setup({ image: "bg-a", layers: { mask_upload: { kind: "mask", value: "mask-upload-1" } }, activeLayer: "mask_upload" })
    expect(stub().dataset.ready).toBe("false")
    await waitFor(() => expect(doc.imageId).toBe("bg-a"))
    expect(stub().dataset.ready).toBe("true")
    await waitFor(() => expect(doc.layer("mask_upload")?.sync.value).toBe("mask-upload-1"))
    const exported = doc.exportLayer("mask_upload")!
    expect(exported.width).toBe(W)
    expect([...exported.data.slice(0, 4)]).toEqual([255, 255, 255, 255])
    // Loaded from its value: nothing to upload again.
    expect(doc.layer("mask_upload")?.sync.pending).toBe(false)
  })

  it("follows a background set from outside (send to inpaint): the old picture's paint goes", async () => {
    const { doc, update } = setup({ image: "bg-a", layers: { mask: { kind: "mask", value: null } }, activeLayer: "mask" })
    await waitFor(() => expect(doc.imageId).toBe("bg-a"))
    doc.polygon("mask", "mask", [0, 0, 4, 0, 4, 4, 0, 4], { color: null, opacity: 1, erase: false })
    expect(doc.isEmpty("mask")).toBe(false)
    update({ image: "bg-sent", layers: { mask: { kind: "mask", value: null } }, activeLayer: "mask" })
    // Until the new picture is in, painting waits (the stub is told the document is not ready).
    expect(stub().dataset.ready).toBe("false")
    await waitFor(() => expect(doc.imageId).toBe("bg-sent"))
    expect(doc.isEmpty("mask")).toBe(true)
    expect(doc.canUndo("mask")).toBe(false)
  })

  it("clears the document when the value's image cannot be loaded, and retries", async () => {
    const { doc, update } = setup({ image: "bg-a", layers: { mask: { kind: "mask", value: null } }, activeLayer: "mask" })
    await waitFor(() => expect(doc.imageId).toBe("bg-a"))
    doc.polygon("mask", "mask", [0, 0, 4, 0, 4, 4, 0, 4], { color: null, opacity: 1, erase: false })
    update({ image: "broken-1", layers: { mask: { kind: "mask", value: null } }, activeLayer: "mask" })
    await waitFor(() => expect(stub().dataset.failed).toBe("true"))
    // The old picture and its mask are not left editable under the new value.
    expect(doc.hasImage).toBe(false)
    expect(doc.imageId).toBeNull()
    expect(stub().dataset.busy).toBe("false")
    expect(stub().dataset.ready).toBe("false")
    const before = loads.calls.length
    fireEvent.click(screen.getByRole("button", { name: "retry" }))
    await waitFor(() => expect(loads.calls.length).toBe(before + 1))
    await waitFor(() => expect(stub().dataset.failed).toBe("true"))
  })

  it("commits painted layers to their params", async () => {
    const { doc, onLayerChange } = setup({ image: "bg-a", layers: { sketch: { kind: "sketch", value: null } }, activeLayer: "sketch" })
    await waitFor(() => expect(doc.imageId).toBe("bg-a"))
    act(() => doc.polygon("sketch", "sketch", [0, 0, 4, 0, 4, 4, 0, 4], { color: [255, 0, 0], opacity: 1, erase: false }))
    await act(() => doc.flush())
    expect(onLayerChange).toHaveBeenCalledWith("sketch", "layer-1")
  })

  it("takes a layer cleared from outside", async () => {
    const { doc, update } = setup({ image: "bg-a", layers: { mask: { kind: "mask", value: "mask-upload-2" } }, activeLayer: "mask" })
    await waitFor(() => expect(doc.layer("mask")?.sync.value).toBe("mask-upload-2"))
    update({ image: "bg-a", layers: { mask: { kind: "mask", value: null } }, activeLayer: "mask" })
    await waitFor(() => expect(doc.isEmpty("mask")).toBe(true))
    expect(doc.layer("mask")?.sync.value).toBeNull()
  })
})
