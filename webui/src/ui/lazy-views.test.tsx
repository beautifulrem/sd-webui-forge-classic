import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { createFormStore } from "./form-store"
import { TabForm } from "./SchemaForm"
import type { Node, OwnerDefaults, TabSchema } from "./schema"

// No preloadViews() here: this file checks the deferred path itself.

function renderNode(node: Node, defaults: OwnerDefaults) {
  const schema: TabSchema = { version: 1, tab: "txt2img", core: { owner: "core", layout: { type: "column", children: [node] }, defaults }, scripts: [] }
  const store = createFormStore(`lazy-${Math.random()}`, { persistDrafts: false })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, enabled: false } } })
  return render(
    <QueryClientProvider client={client}>
      <TabForm schema={schema} formId="txt2img" store={store} />
    </QueryClientProvider>,
  )
}

const slow = { timeout: 15_000 }

describe("deferred node views", { timeout: 30_000 }, () => {
  it("show a placeholder until their code has loaded, then render at once on later mounts", async () => {
    const { container } = renderNode({ type: "color", name: "tint", label: "Tint" }, { tint: "#ff0000" })
    // first render: the view's chunk is still loading
    expect(screen.queryByLabelText("Pick a color for Tint")).toBeNull()
    expect(container.querySelector('[data-slot="skeleton"]')).not.toBeNull()
    expect(await screen.findByLabelText("Pick a color for Tint", {}, slow)).toHaveAttribute("type", "color")
    cleanup()
    // loaded once: a new form renders it without the placeholder
    renderNode({ type: "color", name: "tint", label: "Tint" }, { tint: "#00ff00" })
    expect(screen.getByLabelText("Pick a color for Tint")).toHaveAttribute("type", "color")
  })

  it("keeps the eagerly rendered kinds synchronous", () => {
    renderNode({ type: "toggle", name: "on", label: "Enable thing" }, { on: false })
    expect(screen.getByRole("switch", { name: "Enable thing" })).toBeInTheDocument()
  })
})
