import { EditorView } from "@codemirror/view"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen } from "@testing-library/react"
import { useState } from "react"
import { describe, expect, it } from "vitest"
import PromptEditor from "./PromptEditor"

function Harness({ initial, onValue }: { initial: string; onValue: (value: string) => void }) {
  const [value, setValue] = useState(initial)
  return (
    <PromptEditor
      value={value}
      ariaLabel="Prompt"
      onChange={(next) => {
        setValue(next)
        onValue(next)
      }}
    />
  )
}

function setup(initial: string) {
  const values: string[] = []
  const client = new QueryClient({ defaultOptions: { queries: { enabled: false } } })
  render(
    <QueryClientProvider client={client}>
      <Harness initial={initial} onValue={(v) => values.push(v)} />
    </QueryClientProvider>,
  )
  const content = screen.getByRole("textbox", { name: "Prompt" })
  const view = EditorView.findFromDOM(content)!
  const key = (k: string, mods: { ctrlKey?: boolean; altKey?: boolean; shiftKey?: boolean } = {}) =>
    fireEvent.keyDown(content, { key: k, ...mods })
  return { view, key, values, doc: () => view.state.doc.toString() }
}

describe("PromptEditor keys", () => {
  it("Ctrl+↑/↓ edits attention and reports the change", () => {
    const { view, key, values, doc } = setup("a cat, blue sky")
    view.dispatch({ selection: { anchor: 13 } })
    key("ArrowUp", { ctrlKey: true })
    expect(doc()).toBe("a cat, blue (sky:1.1)")
    key("ArrowDown", { ctrlKey: true })
    expect(doc()).toBe("a cat, blue sky")
    expect(values.at(-1)).toBe("a cat, blue sky")
  })

  it("Alt+←/→ moves tags", () => {
    const { view, key, doc } = setup("a, b, c")
    view.dispatch({ selection: { anchor: 4 } })
    key("ArrowRight", { altKey: true })
    expect(doc()).toBe("a, c, b")
  })

  it("undo restores the previous text", () => {
    const { view, key, doc } = setup("sky")
    view.dispatch({ selection: { anchor: 1 } })
    key("ArrowUp", { ctrlKey: true })
    expect(doc()).toBe("(sky:1.1)")
    key("z", { ctrlKey: true })
    expect(doc()).toBe("sky")
  })
})
