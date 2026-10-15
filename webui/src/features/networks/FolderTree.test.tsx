import { fireEvent, render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import type { NetworkFolder } from "@/api/types"
import { FolderTree } from "./FolderTree"

const FOLDERS: NetworkFolder[] = [
  { path: "anime", name: "anime", count: 3, children: [{ path: "anime/girls", name: "girls", count: 2, children: [] }] },
  { path: "styles", name: "styles", count: 4, children: [] },
]

describe("folder tree", () => {
  it("is driven by the keyboard like a WAI-ARIA tree", () => {
    const onSelect = vi.fn()
    render(<FolderTree folders={FOLDERS} total={9} selected="" onSelect={onSelect} />)
    const tree = screen.getByRole("tree", { name: "Folders" })
    const rows = () => within(tree).getAllByRole("treeitem")
    expect(rows().map((row) => row.textContent)).toEqual(["All9", "anime3", "styles4"])
    // one tab stop: the selected row
    expect(rows().filter((row) => row.tabIndex === 0)).toEqual([rows()[0]])

    rows()[0]!.focus()
    fireEvent.keyDown(tree, { key: "ArrowDown" })
    expect(document.activeElement).toBe(rows()[1])
    expect(rows()[1]).toHaveAttribute("aria-expanded", "false")
    fireEvent.keyDown(tree, { key: "ArrowRight" })
    expect(rows()[1]).toHaveAttribute("aria-expanded", "true")
    expect(rows().map((row) => row.textContent)).toEqual(["All9", "anime3", "girls2", "styles4"])
    fireEvent.keyDown(tree, { key: "ArrowRight" })
    expect(document.activeElement).toHaveTextContent("girls")
    fireEvent.keyDown(tree, { key: "Enter" })
    expect(onSelect).toHaveBeenLastCalledWith("anime/girls")
    fireEvent.keyDown(tree, { key: "ArrowLeft" })
    expect(document.activeElement).toHaveTextContent("anime")
    fireEvent.keyDown(tree, { key: "ArrowLeft" })
    expect(rows()[1]).toHaveAttribute("aria-expanded", "false")
    fireEvent.keyDown(tree, { key: "End" })
    expect(document.activeElement).toHaveTextContent("styles")
    fireEvent.keyDown(tree, { key: " " })
    expect(onSelect).toHaveBeenLastCalledWith("styles")
    fireEvent.keyDown(tree, { key: "Home" })
    fireEvent.keyDown(tree, { key: "Enter" })
    expect(onSelect).toHaveBeenLastCalledWith("")
  })

  it("reveals a folder chosen elsewhere", () => {
    const { rerender } = render(<FolderTree folders={FOLDERS} total={9} selected="" onSelect={() => {}} />)
    rerender(<FolderTree folders={FOLDERS} total={9} selected="anime/girls" onSelect={() => {}} />)
    const girls = screen.getByRole("treeitem", { name: /girls/ })
    expect(girls).toHaveAttribute("aria-selected", "true")
    expect(girls.tabIndex).toBe(0)
  })
})
