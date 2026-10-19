import { act, render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import type { EditorDocument } from "../engine/document"
import { useDocValue, useLayerPending } from "./use-doc"

/** A listener set, as the document and a layer's sync have. */
class Emitter {
  readonly listeners = new Set<() => void>()
  subscribe(listener: () => void) {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }
  emit() {
    for (const listener of [...this.listeners]) listener()
  }
}

class FakeSync extends Emitter {
  pending = false
}

/** The parts of `EditorDocument` the hooks use: subscribe, layer(key).sync, a value to select. */
class FakeDoc extends Emitter {
  zoom = 1
  readonly layers = new Map<string, { sync: FakeSync }>()
  layer(key: string) {
    return this.layers.get(key) ?? null
  }
  addLayer(key: string) {
    const sync = new FakeSync()
    this.layers.set(key, { sync })
    this.emit()
    return sync
  }
  asDoc() {
    return this as unknown as EditorDocument
  }
}

let renders = 0

function Zoom({ doc }: { doc: FakeDoc }) {
  renders++
  const zoom = useDocValue(doc.asDoc(), (d) => (d as unknown as FakeDoc).zoom)
  return <output aria-label="zoom">{zoom}</output>
}

function Pending({ doc, layer }: { doc: FakeDoc; layer: string | null }) {
  const pending = useLayerPending(doc.asDoc(), layer)
  return <output aria-label="pending">{String(pending)}</output>
}

describe("useDocValue", () => {
  it("re-renders only when the selected value changes", () => {
    const doc = new FakeDoc()
    renders = 0
    render(<Zoom doc={doc} />)
    expect(screen.getByLabelText("zoom")).toHaveTextContent("1")
    const first = renders

    act(() => doc.emit()) // a stroke: the document changed, the zoom did not
    expect(renders).toBe(first)

    act(() => {
      doc.zoom = 2
      doc.emit()
    })
    expect(screen.getByLabelText("zoom")).toHaveTextContent("2")
    expect(renders).toBe(first + 1)
  })
})

describe("useLayerPending", () => {
  it("is false without a layer key", () => {
    render(<Pending doc={new FakeDoc()} layer={null} />)
    expect(screen.getByLabelText("pending")).toHaveTextContent("false")
  })

  it("follows a layer that appears on the first edit and its upload state", () => {
    const doc = new FakeDoc()
    render(<Pending doc={doc} layer="mask" />)
    expect(screen.getByLabelText("pending")).toHaveTextContent("false")

    let sync!: FakeSync
    act(() => {
      sync = doc.addLayer("mask")
    })
    expect(sync.listeners.size).toBe(1) // bound to the new layer's sync

    act(() => {
      sync.pending = true
      sync.emit()
    })
    expect(screen.getByLabelText("pending")).toHaveTextContent("true")

    act(() => {
      sync.pending = false
      sync.emit()
    })
    expect(screen.getByLabelText("pending")).toHaveTextContent("false")

    // Further document events keep the one subscription.
    act(() => doc.emit())
    expect(sync.listeners.size).toBe(1)
  })

  it("rebinds when the layer is replaced and unsubscribes on unmount", () => {
    const doc = new FakeDoc()
    const first = doc.addLayer("mask")
    const view = render(<Pending doc={doc} layer="mask" />)
    expect(first.listeners.size).toBe(1)

    let second!: FakeSync
    act(() => {
      second = doc.addLayer("mask")
    })
    expect(first.listeners.size).toBe(0)
    expect(second.listeners.size).toBe(1)

    act(() => {
      doc.layers.delete("mask")
      doc.emit()
    })
    expect(second.listeners.size).toBe(0)

    view.unmount()
    expect(doc.listeners.size).toBe(0)
  })
})
