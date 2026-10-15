import { fireEvent, render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import type { NetworkItem, NetworkKind } from "@/api/types"
import { ItemMenu } from "./ItemMenu"

const KIND: NetworkKind = { id: "lora", title: "LoRA", action: "prompt", negative: true, fields: ["description"], builtin: true }
const item = (extra: Partial<NetworkItem> = {}): NetworkItem => ({ name: "ink", title: "ink", kind: "lora", path: "ink.safetensors", folder: "", tags: [], search_terms: [], editable: true, hidden: false, ...extra })

function open(props: Partial<Parameters<typeof ItemMenu>[0]> = {}) {
  const anchor = document.createElement("div")
  document.body.append(anchor)
  const onCommand = vi.fn()
  render(<ItemMenu item={item()} kind={KIND} anchor={anchor} active={false} canOpenFolder onClose={() => {}} onCommand={onCommand} {...props} />)
  return { menu: screen.getByRole("menu"), onCommand }
}

const labels = (menu: HTMLElement) => within(menu).getAllByRole("menuitem").map((entry) => entry.textContent?.replace(/[IE]$/, ""))

describe("item menu", () => {
  it("offers every command of an item with a file", () => {
    const { menu, onCommand } = open()
    expect(labels(menu)).toEqual(["Add to the prompt", "Metadata", "Edit metadata", "Set preview", "Copy path", "Open folder"])
    fireEvent.click(within(menu).getByRole("menuitem", { name: /Metadata/ }))
    expect(onCommand).toHaveBeenCalledWith("metadata", expect.objectContaining({ name: "ink" }))
  })

  it("leaves out what needs a model file, and Open folder when folders are hidden", () => {
    expect(labels(open({ item: item({ editable: false }) }).menu)).toEqual(["Add to the prompt", "Metadata", "Copy path"])
  })

  it("hides Open folder when the server does not allow it, and says Remove for an active card", () => {
    expect(labels(open({ canOpenFolder: false, active: true }).menu)).toEqual(["Remove from the prompt", "Metadata", "Edit metadata", "Set preview", "Copy path"])
  })

  it("opens at the pointer for a right click", () => {
    const element = document.createElement("div")
    document.body.append(element)
    const { menu } = open({ anchor: { x: 120, y: 80, element } })
    expect(menu).toBeInTheDocument()
  })
})
