/** Inputs without a schema label are named after their section or group, not their param name. */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { createFormStore } from "./form-store"
import { readableName } from "./fallback-name"
import { TabForm } from "./SchemaForm"
import type { Node, OwnerDefaults } from "./schema"

function renderLayout(layout: Node, defaults: OwnerDefaults) {
  const store = createFormStore(`test-${Math.random()}`, { persistDrafts: false })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, enabled: false } } })
  render(
    <QueryClientProvider client={client}>
      <TabForm schema={{ version: 1, tab: "txt2img", core: { owner: "core", layout, defaults }, scripts: [] }} formId="txt2img" store={store} />
    </QueryClientProvider>,
  )
}

describe("fallback accessible names", () => {
  it("uses the nearest section or group title", () => {
    renderLayout(
      {
        type: "column",
        children: [
          { type: "section", title: "Anchor prompt", open: true, children: [{ type: "text", name: "anchor", label: "", lines: 2 }] },
          { type: "group", title: "LoRA Replacements", open: true, children: [{ type: "text", name: "replacements", label: "", lines: 4 }] },
          { type: "toggle", name: "keep_it", label: "" },
        ],
      } as unknown as Node,
      { anchor: "", replacements: "", keep_it: false },
    )
    expect(screen.getByRole("textbox", { name: "Anchor prompt" })).toBeInTheDocument()
    expect(screen.getByRole("textbox", { name: "LoRA Replacements" })).toBeInTheDocument()
    expect(screen.getByRole("switch", { name: "Keep it" })).toBeInTheDocument()
    expect(screen.queryByRole("textbox", { name: "anchor" })).not.toBeInTheDocument()
  })

  it("makes a param name readable as the last resort", () => {
    expect(readableName("extra_options")).toBe("Extra options")
    expect(readableName("user-id")).toBe("User id")
  })
})
