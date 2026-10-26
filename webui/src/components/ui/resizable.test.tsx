import { act, fireEvent, render, renderHook, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { type PanelImperativeHandle, ResizableHandle, ResizablePanel, ResizablePanelGroup, useDefaultLayout, usePanelRef } from "./resizable"

/** jsdom has no layout: the group is 1,000 px wide. */
beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockImplementation(function (this: HTMLElement) {
    return this.dataset.slot === "resizable-panel-group" ? 1000 : 0
  })
})
afterEach(() => vi.restoreAllMocks())

const sizes = () => ["a", "b", "c"].map((id) => Number(document.getElementById(id)?.style.flexGrow))

function Panes({ onLayoutChanged, onResize, refs }: { onLayoutChanged?: (layout: Record<string, number>) => void; onResize?: (id: string, px: number) => void; refs?: { a?: React.Ref<PanelImperativeHandle | null> } }) {
  return (
    <ResizablePanelGroup orientation="horizontal" onLayoutChanged={onLayoutChanged}>
      {/* 20 % to 50 %, collapsible */}
      <ResizablePanel id="a" panelRef={refs?.a} defaultSize="30" minSize={200} maxSize="50" collapsible collapsedSize={0} onResize={(size) => onResize?.("a", size.inPixels)}>
        A
      </ResizablePanel>
      <ResizableHandle aria-label="Resize A" />
      {/* at least 300 px */}
      <ResizablePanel id="b" minSize={300}>
        B
      </ResizablePanel>
      <ResizableHandle aria-label="Resize C" />
      <ResizablePanel id="c" defaultSize="20" minSize={100} maxSize="35">
        C
      </ResizablePanel>
    </ResizablePanelGroup>
  )
}

describe("resizable panels", () => {
  it("lays the panels out from their default sizes, the flexible one taking the rest", () => {
    render(<Panes />)
    expect(sizes()).toEqual([30, 50, 20])
    const handle = screen.getByRole("separator", { name: "Resize A" })
    expect(handle).toHaveAttribute("aria-orientation", "vertical")
    expect(handle).toHaveAttribute("aria-controls", "a")
    expect(handle).toHaveAttribute("aria-valuenow", "30")
    expect(handle).toHaveAttribute("aria-valuemin", "0")
    expect(handle).toHaveAttribute("aria-valuemax", "50")
  })

  it("moves a separator with the keyboard within the panels' limits and collapses with Enter", () => {
    const onLayoutChanged = vi.fn()
    render(<Panes onLayoutChanged={onLayoutChanged} />)
    const handle = screen.getByRole("separator", { name: "Resize A" })
    fireEvent.keyDown(handle, { key: "ArrowRight" })
    expect(sizes()).toEqual([35, 45, 20])
    expect(onLayoutChanged).toHaveBeenLastCalledWith({ a: 35, b: 45, c: 20 })
    fireEvent.keyDown(handle, { key: "End" })
    // A stops at its maximum (B would still have room)
    expect(sizes()).toEqual([50, 30, 20])
    fireEvent.keyDown(handle, { key: "Enter" })
    expect(sizes()).toEqual([0, 80, 20])
    fireEvent.keyDown(handle, { key: "Enter" })
    // expanded to the size it had
    expect(sizes()).toEqual([50, 30, 20])
    // B cannot go below 300 px: C's separator stops there
    const second = screen.getByRole("separator", { name: "Resize C" })
    fireEvent.keyDown(second, { key: "Home" })
    expect(sizes()).toEqual([50, 30, 20])
    fireEvent.keyDown(second, { key: "ArrowRight" })
    expect(sizes()).toEqual([50, 35, 15])
  })

  it("moves the focus between separators with F6", () => {
    render(<Panes />)
    const [first, second] = screen.getAllByRole("separator")
    act(() => first!.focus())
    fireEvent.keyDown(first!, { key: "F6" })
    expect(second).toHaveFocus()
    fireEvent.keyDown(second!, { key: "F6" })
    expect(first).toHaveFocus()
  })

  it("resizes by dragging, collapsing a collapsible panel dragged well below its minimum", () => {
    const onLayoutChanged = vi.fn()
    render(<Panes onLayoutChanged={onLayoutChanged} />)
    const handle = screen.getByRole("separator", { name: "Resize A" })
    fireEvent.pointerDown(handle, { clientX: 300, button: 0, pointerId: 1 })
    fireEvent.pointerMove(handle, { clientX: 400, pointerId: 1 })
    expect(sizes()).toEqual([40, 40, 20])
    // below the minimum (20 %) but above halfway to collapsed: held at the minimum
    fireEvent.pointerMove(handle, { clientX: 150, pointerId: 1 })
    expect(sizes()).toEqual([20, 60, 20])
    fireEvent.pointerMove(handle, { clientX: 50, pointerId: 1 })
    expect(sizes()).toEqual([0, 80, 20])
    expect(onLayoutChanged).not.toHaveBeenCalled()
    fireEvent.pointerUp(handle, { pointerId: 1 })
    expect(onLayoutChanged).toHaveBeenCalledWith({ a: 0, b: 80, c: 20 })
  })

  it("collapses and expands a panel through its ref, giving the room to its neighbour", () => {
    const onResize = vi.fn()
    const { result } = renderHook(() => usePanelRef())
    render(<Panes onResize={onResize} refs={{ a: result.current }} />)
    expect(onResize).toHaveBeenLastCalledWith("a", 300)
    act(() => result.current.current!.collapse())
    expect(sizes()).toEqual([0, 80, 20])
    expect(result.current.current!.isCollapsed()).toBe(true)
    expect(onResize).toHaveBeenLastCalledWith("a", 0)
    act(() => result.current.current!.expand())
    expect(sizes()).toEqual([30, 50, 20])
    expect(result.current.current!.getSize()).toEqual({ asPercentage: 30, inPixels: 300 })
  })

  it("starts from a kept layout when it names the same panels", () => {
    render(
      <ResizablePanelGroup defaultLayout={{ a: 40, b: 60 }}>
        <ResizablePanel id="a" defaultSize="30">
          A
        </ResizablePanel>
        <ResizableHandle />
        <ResizablePanel id="b">B</ResizablePanel>
      </ResizablePanelGroup>,
    )
    expect([document.getElementById("a")!.style.flexGrow, document.getElementById("b")!.style.flexGrow]).toEqual(["40", "60"])
  })
})

describe("useDefaultLayout", () => {
  it("keeps the layout under react-resizable-panels' key and format", () => {
    localStorage.setItem("react-resizable-panels:page:a:b", JSON.stringify({ a: 25, b: 75 }))
    const { result } = renderHook(() => useDefaultLayout({ id: "page", panelIds: ["a", "b"] }))
    expect(result.current.defaultLayout).toEqual({ a: 25, b: 75 })
    result.current.onLayoutChanged({ a: 40, b: 60 })
    expect(JSON.parse(localStorage.getItem("react-resizable-panels:page:a:b")!)).toEqual({ a: 40, b: 60 })
  })

  it("ignores a damaged entry", () => {
    localStorage.setItem("react-resizable-panels:page:a:b", "{not json")
    const { result } = renderHook(() => useDefaultLayout({ id: "page", panelIds: ["a", "b"] }))
    expect(result.current.defaultLayout).toBeUndefined()
  })
})
