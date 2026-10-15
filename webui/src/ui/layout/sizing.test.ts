import { describe, expect, it } from "vitest"
import type { Node } from "../schema"
import { cellProps } from "./sizing"

describe("cellProps", () => {
  it("bottom-aligns controls and top-aligns containers in a row", () => {
    expect(cellProps({ type: "button", label: "Go", action: "go", variant: "primary" } as Node).className).toBe("remi-cell remi-cell-content")
    expect(cellProps({ type: "text", name: "a", label: "A", default: "" } as Node).className).toBe("remi-cell remi-cell-auto")
    expect(cellProps({ type: "column", children: [] } as unknown as Node).className).toBe("remi-cell remi-cell-auto remi-cell-top")
    const span = cellProps({ type: "group", width: 6, children: [] } as unknown as Node)
    expect(span.className).toBe("remi-cell remi-cell-span remi-cell-top")
    expect(span.style).toEqual({ "--span": 6 })
  })
})
