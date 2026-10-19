/** The `prompt_before` slot: scripts placed above the main prompt (Prompt Anchor), rendered once. */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, within } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { PROMPT_BEFORE_SLOT } from "@/ui/form-index"
import type { Node, ScriptSchema, TabSchema } from "@/ui/schema"
import { curateCore } from "./curate"
import { ParamsPane } from "./ParamsPane"

const DOCUMENT_POSITION_FOLLOWING = 4

// Like the real Prompt Anchor entry (section "prompt", slot "prompt_before" on the server).
const ANCHOR: ScriptSchema = {
  id: "prompt_anchor",
  title: "Prompt Anchor",
  kind: "alwayson",
  section: PROMPT_BEFORE_SLOT,
  order: 0,
  layout: {
    type: "section",
    title: "Anchor prompt",
    open: true,
    enable: { type: "toggle", name: "enabled", label: "Anchor prompt (prepended to positive prompt at generation)" },
    children: [{ type: "text", name: "anchor", label: "", lines: 2, prompt: true, placeholder: "Anchor text gets prepended" }],
  },
  defaults: { enabled: true, anchor: "" },
  args: ["enabled", "anchor"],
}

/** The mock txt2img schema with the core prompt group as the server builds it: the slot first. */
function withAnchor(): TabSchema {
  const schema = tabSchema("txt2img")
  const layout = schema.core.layout
  if (layout.type !== "column") throw new Error("expected a column")
  const [prompt, negative, styles, ...rest] = layout.children
  const group: Node = { type: "group", children: [{ type: "slot", section: PROMPT_BEFORE_SLOT }, prompt!, negative!, styles!] }
  return { ...schema, core: { ...schema.core, layout: { ...layout, children: [group, ...rest] } }, scripts: [ANCHOR, ...schema.scripts] }
}

describe("prompt_before slot", () => {
  it("is taken out of the curated blocks", () => {
    const curated = curateCore(withAnchor().core.layout)
    expect(curated.prompt?.name).toBe("prompt")
    const slots: string[] = []
    const visit = (node: Node) => {
      if (node.type === "slot") slots.push(node.section)
      if ("children" in node) for (const child of node.children as Node[]) visit(child)
    }
    for (const block of curated.blocks) visit(block.node)
    expect(slots).not.toContain(PROMPT_BEFORE_SLOT)
  })

  it("renders its scripts once, directly above the main prompt", () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, enabled: false } } })
    render(
      <QueryClientProvider client={client}>
        <ParamsPane schema={withAnchor()} tab="txt2img" />
      </QueryClientProvider>,
    )
    const anchors = document.querySelectorAll("[data-script=prompt_anchor]")
    expect(anchors).toHaveLength(1)
    const anchor = anchors[0]!
    expect(anchor.closest("[data-slot-section]")).toHaveAttribute("data-slot-section", PROMPT_BEFORE_SLOT)
    const prompts = screen.getByRole("region", { name: "Prompts" })
    expect(prompts.contains(anchor)).toBe(true)
    // The main prompt follows it (document order)
    const promptLabel = within(prompts).getByText("Prompt", { selector: "label" })
    expect(anchor.compareDocumentPosition(promptLabel) & DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(within(anchor as HTMLElement).getByRole("switch", { name: "Enable Anchor prompt" })).toBeChecked()
  })
})
