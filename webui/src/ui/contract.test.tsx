/**
 * Contract tests against the real backend schemas (`api/mock/contract/*.json`,
 * dumped from `remi.core.tabs.tab_schema()` by `tools/dump_schemas.py`).
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import extras from "@/api/mock/contract/extras.json"
import img2img from "@/api/mock/contract/img2img.json"
import txt2img from "@/api/mock/contract/txt2img.json"
import { buildIndex, inputsIn } from "./form-index"
import { createFormStore } from "./form-store"
import { TabForm } from "./SchemaForm"
import { isInput, type TabSchema } from "./schema"
import { jobRequest, tabOwners } from "./tab"

const SCHEMAS = { txt2img, img2img, extras } as unknown as Record<string, TabSchema>

afterEach(() => vi.restoreAllMocks())

describe.each(Object.entries(SCHEMAS))("real %s schema", (tab, schema) => {
  it("renders without errors and round-trips every default", () => {
    const errors = vi.spyOn(console, "error")
    const warnings = vi.spyOn(console, "warn")
    const store = createFormStore(`contract-${tab}`, { persistDrafts: false })
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { enabled: false } } })}>
        <TabForm schema={schema} store={store} />
      </QueryClientProvider>,
    )
    expect(errors).not.toHaveBeenCalled()
    expect(warnings).not.toHaveBeenCalled()

    const body = jobRequest(store, schema)
    expect(body.tab).toBe(tab)
    expect(body.core).toEqual({ ...schema.core.defaults, ...body.core })
    for (const [name, value] of Object.entries(schema.core.defaults)) expect(body.core[name]).toEqual(value)
    for (const script of schema.scripts) {
      for (const [name, value] of Object.entries(script.defaults)) expect(body.scripts[script.id]?.[name]).toEqual(value)
    }
  })

  it("uses only node kinds and keys the renderer knows", () => {
    const index = buildIndex(tabOwners(schema))
    for (const [owner, inputs] of index.inputs) {
      for (const [name, node] of inputs) {
        expect(isInput(node), `${owner}.${name}`).toBe(true)
        expect(node.label, `${owner}.${name} label`).toBeTypeOf("string")
      }
    }
    // Conflict params name existing inputs.
    for (const ref of schema.conflict_params ?? []) {
      const [owner, name] = ref.split(".", 2) as [string, string]
      expect(index.inputs.get(owner)?.has(name), ref).toBe(true)
    }
    // Every script param with a default has an input.
    for (const script of schema.scripts) {
      const names = new Set(inputsIn(script.layout).map((n) => n.name))
      for (const name of Object.keys(script.defaults)) expect(names.has(name), `${script.id}.${name}`).toBe(true)
    }
  })
})

describe("real txt2img schema", () => {
  it("places the sampler and seed scripts and loads the conflict params", () => {
    const store = createFormStore("contract-slots", { persistDrafts: false })
    render(<TabForm schema={SCHEMAS.txt2img!} store={store} />)
    expect(document.querySelector("[data-slot-section=sampler] [data-script=sampler]")).not.toBeNull()
    expect(document.querySelector("[data-slot-section=seed] [data-script=seed]")).not.toBeNull()
    expect(store.getState().conflictParams).toEqual(new Set(SCHEMAS.txt2img!.conflict_params))
    expect(screen.getByRole("switch", { name: "Enable Hires. fix" })).toBeInTheDocument()
  })
})
