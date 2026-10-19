/** The img2img stage's own logic (view switch, keyboard, batch modes), its editor and bindings stubbed out. */
import { fireEvent, render, screen } from "@testing-library/react"
import { useState } from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { ImageEditorProps } from "../types"

type Binding = { background: string; layers: { param: string }[] } | null

const state = vi.hoisted(() => ({
  binding: null as Binding,
  canvas: true,
  editorProps: [] as Partial<ImageEditorProps>[],
  stageArgs: [] as (string[] | undefined)[],
}))

const img2img = vi.hoisted(() => ({
  setCoreValues: vi.fn(),
  setSizeFromImage: vi.fn(),
  setSourceImage: vi.fn(() => Promise.resolve()),
}))

const preload = vi.hoisted(() => ({ preloadImageEditor: vi.fn(() => Promise.resolve()) }))

vi.mock("./use-img2img", () => ({
  IMG2IMG: "img2img",
  ...img2img,
  useImg2imgBinding: () => state.binding,
  useModeLayers: () => ({ canvas: state.canvas }),
  useImg2imgEditorProps: () => ({ image: "upload-1" }),
}))

vi.mock("./stage-view", () => ({
  useImg2imgStageView: (params: string[] | undefined) => {
    state.stageArgs.push(params)
    return useState<"source" | "result">("source")
  },
}))

vi.mock("../preload", () => preload)

vi.mock("../ImageEditor", () => ({
  ImageEditor: (props: ImageEditorProps) => {
    state.editorProps.push(props)
    return <div aria-label={props.label}>editor</div>
  },
}))

const { Img2imgStage } = await import("./Img2imgStage")

function renderStage() {
  return render(<Img2imgStage result={<p>results</p>} actions={<button type="button">Send to</button>} leading={<span>lead</span>} trailing={<span>trail</span>} />)
}

describe("Img2imgStage", () => {
  beforeEach(() => {
    state.binding = { background: "init_img", layers: [{ param: "mask" }, { param: "sketch" }] }
    state.canvas = true
    state.editorProps = []
    state.stageArgs = []
    for (const fn of [...Object.values(img2img), preload.preloadImageEditor]) fn.mockClear()
  })

  it("shows the source editor first and preloads the editor chunk", () => {
    renderStage()
    expect(preload.preloadImageEditor).toHaveBeenCalledOnce()
    expect(state.stageArgs[0]).toEqual(["init_img", "mask", "sketch"])
    const source = screen.getByRole("tab", { name: "Source" })
    expect(source).toHaveAttribute("aria-selected", "true")
    expect(source).toHaveAttribute("aria-controls", "img2img-stage-source")
    expect(screen.getByRole("tab", { name: "Result" })).toHaveAttribute("tabindex", "-1")
    expect(screen.getByRole("tabpanel")).toHaveAttribute("id", "img2img-stage-source")
    expect(screen.getByLabelText("Source image")).toBeInTheDocument()
    // The result actions belong to the result view.
    expect(screen.queryByRole("button", { name: "Send to" })).toBeNull()
    expect(screen.getByText("lead")).toBeInTheDocument()
    expect(screen.getByText("trail")).toBeInTheDocument()
  })

  it("switches views by click and with the arrow keys, moving the focus", () => {
    renderStage()
    fireEvent.click(screen.getByRole("tab", { name: "Result" }))
    expect(screen.getByRole("tabpanel")).toHaveAttribute("id", "img2img-stage-result")
    expect(screen.getByText("results")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Send to" })).toBeInTheDocument()

    fireEvent.keyDown(screen.getByRole("tab", { name: "Result" }), { key: "ArrowLeft" })
    expect(screen.getByRole("tab", { name: "Source" })).toHaveAttribute("aria-selected", "true")
    expect(screen.getByRole("tab", { name: "Source" })).toHaveFocus()

    fireEvent.keyDown(screen.getByRole("tab", { name: "Source" }), { key: "ArrowRight" })
    expect(screen.getByRole("tab", { name: "Result" })).toHaveAttribute("aria-selected", "true")
    expect(screen.getByRole("tab", { name: "Result" })).toHaveFocus()

    // Other keys leave the view alone.
    fireEvent.keyDown(screen.getByRole("tab", { name: "Result" }), { key: "Enter" })
    expect(screen.getByRole("tab", { name: "Result" })).toHaveAttribute("aria-selected", "true")
  })

  it("writes the editor's changes to the img2img form", () => {
    renderStage()
    const props = state.editorProps.at(-1)!
    expect(props).toMatchObject({ docKey: "img2img", pendingKey: "img2img", image: "upload-1", fill: true })
    props.onImageChange?.({ id: "upload-2", width: 640, height: 960 })
    expect(img2img.setSourceImage).toHaveBeenCalledWith(state.binding, "upload-2", { id: "upload-2", width: 640, height: 960 })
    props.onImageChange?.(null)
    expect(img2img.setSourceImage).toHaveBeenLastCalledWith(state.binding, null, null)
    props.onLayerChange?.("mask", "upload-3")
    expect(img2img.setCoreValues).toHaveBeenCalledWith({ mask: "upload-3" })
    props.onUseImageSize?.({ width: 512, height: 768 })
    expect(img2img.setSizeFromImage).toHaveBeenCalledWith({ width: 512, height: 768 })
  })

  it("shows only the results in a batch mode (no canvas)", () => {
    state.canvas = false
    renderStage()
    expect(screen.queryByRole("tablist")).toBeNull()
    expect(screen.queryByRole("tabpanel")).toBeNull()
    expect(screen.getByText("results")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Send to" })).toBeInTheDocument()
    expect(state.editorProps).toHaveLength(0)
  })

  it("shows the results while the form has no editor binding yet", () => {
    state.binding = null
    renderStage()
    expect(state.stageArgs[0]).toBeUndefined()
    expect(screen.queryByRole("tablist")).toBeNull()
    expect(screen.getByText("results")).toBeInTheDocument()
  })
})
