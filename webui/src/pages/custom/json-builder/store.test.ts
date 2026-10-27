import { beforeEach, describe, expect, it } from "vitest"
import { decompilePrompt } from "./decompile"
import { restore, useBuilder } from "./store"
import { COMPILED } from "./testdata"

const initial = useBuilder.getState()

beforeEach(() => useBuilder.setState(initial, true))

describe("builder store", () => {
  it("starts with one empty object item on a 1024 × 1024 canvas", () => {
    const s = useBuilder.getState()
    expect([s.width, s.height, s.canvas]).toEqual([1024, 1024, "single"])
    expect(s.items).toEqual([{ id: 1, label: "object", caption: "", text: "", bbox: [102, 102, 409, 409] }])
  })

  it("adds, edits and removes items; removing the last one leaves an empty item", () => {
    const { addItem, updateItem, removeItem } = useBuilder.getState()
    addItem()
    updateItem(2, { label: "text", text: "hi" })
    expect(useBuilder.getState().items.map((item) => [item.id, item.label])).toEqual([
      [1, "object"],
      [2, "text"],
    ])
    removeItem(1)
    removeItem(2)
    expect(useBuilder.getState().items.map((item) => item.id)).toEqual([3])
    useBuilder.getState().clearItems()
    expect(useBuilder.getState().items.map((item) => item.id)).toEqual([4])
  })

  it("clamps the boxes into a new size only when asked", () => {
    useBuilder.getState().setBox(1, [800, 800, 1000, 1000])
    useBuilder.getState().setSize(512, 512)
    expect(useBuilder.getState().items[0]!.bbox).toEqual([800, 800, 1000, 1000])
    useBuilder.getState().setSize(512, 512, true)
    expect(useBuilder.getState().items[0]!.bbox).toEqual([312, 312, 512, 512])
  })

  it("loads a draft with ids after the last one", () => {
    useBuilder.getState().set({ extra: "<lora:x:1>" })
    useBuilder.getState().load(decompilePrompt(COMPILED, 1024, 768))
    const s = useBuilder.getState()
    expect([s.width, s.height, s.extra, s.lastId]).toEqual([1024, 768, "", 3])
    expect(s.items.map((item) => item.id)).toEqual([2, 3])
  })

  it("is kept in the browser", () => {
    useBuilder.getState().set({ caption: "kept" })
    expect(JSON.parse(localStorage.getItem("remi-json-builder")!).state).toMatchObject({ caption: "kept", width: 1024 })
  })
})

describe("restore", () => {
  it("keeps only the well-formed parts of a stored draft", () => {
    expect(restore(null)).toEqual({})
    expect(restore({ width: -1, height: 512, caption: 3, canvas: "grid", ar: "7:5", mp: 0, items: "x" })).toEqual({})
    const item = { id: 4, label: "object", caption: "c", text: "", bbox: [0, 0, 16, 16] }
    expect(restore({ width: 512, height: 768, canvas: "collage", ar: "16:9", mp: 2, items: [item, { id: 5, label: "x" }], lastId: 2 })).toEqual({
      width: 512,
      height: 768,
      canvas: "collage",
      ar: "16:9",
      mp: 2,
      items: [item],
      lastId: 4,
    })
    // duplicate ids would break the list keys: the stored items are dropped
    expect(restore({ items: [item, item] })).toEqual({})
  })
})
