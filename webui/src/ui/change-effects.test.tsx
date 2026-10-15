/** Actions wired to edits: on_change (debounced), conflict locks, scripts context, bulk resets, __settings__, stale drops. */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { keys, queryClient } from "@/api/queries"
import type { Bootstrap } from "@/api/types"
import { actionRequest, resetOwners, runAction } from "./actions"
import { createFormStore, type FormStore } from "./form-store"
import { TabForm } from "./SchemaForm"
import type { TabSchema } from "./schema"

type Call = { path: string; method: string; body: Record<string, unknown> }

/** Routes fetch by path; records every call with its decoded JSON body. */
function mockServer(routes: Record<string, unknown | ((body: Record<string, unknown>) => unknown)> = {}) {
  const calls: Call[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const request = input as Request
    const path = new URL(request.url).pathname.replace("/api/v2", "")
    const body = request.method === "GET" ? {} : ((await request.json()) as Record<string, unknown>)
    calls.push({ path, method: request.method, body })
    const route = routes[path]
    const data = typeof route === "function" ? route(body) : (route ?? {})
    return new Response(JSON.stringify(data), { headers: { "Content-Type": "application/json" } })
  })
  return calls
}

const SCHEMA: TabSchema = {
  version: 1,
  tab: "txt2img",
  core: {
    owner: "core",
    layout: {
      type: "column",
      children: [
        {
          type: "section",
          title: "Hires fix",
          enable: { type: "toggle", name: "enable_hr", label: "Hires fix", exclusive: { group: "g", member: "hires" }, on_change: "hr_resolution" },
          children: [
            { type: "slider", name: "hr_scale", label: "Upscale by", minimum: 1, maximum: 4, step: 0.5, on_change: "hr_resolution" },
            { type: "output", name: "hr_final_resolution", kind: "text" },
          ],
        },
        { type: "slot", section: "sampler" },
      ],
    },
    defaults: { enable_hr: false, hr_scale: 2 },
  },
  scripts: [
    {
      id: "sampler",
      title: "Sampler",
      kind: "alwayson",
      section: "sampler",
      layout: {
        type: "choice",
        name: "sampler_name",
        label: "Sampler",
        style: "radio",
        choices: [
          { value: "Euler", label: "Euler" },
          { value: "LCM", label: "LCM" },
        ],
      },
      defaults: { sampler_name: "Euler" },
    },
  ],
  conflict_params: ["core.enable_hr", "sampler.sampler_name"],
}

function renderForm(schema: TabSchema = SCHEMA): FormStore {
  const store = createFormStore(`fx-${Math.random()}`, { persistDrafts: false })
  render(
    <QueryClientProvider client={new QueryClient()}>
      <TabForm schema={schema} formId="txt2img" store={store} />
    </QueryClientProvider>,
  )
  return store
}

afterEach(() => vi.restoreAllMocks())

describe("action requests", () => {
  it("send the core values and every other script's values", () => {
    const store = createFormStore("ar", { persistDrafts: false })
    store.setState({ values: { core: { a: 1 }, sampler: { s: 2 }, seed: { seed: 3 } } })
    expect(actionRequest({ store, tab: "txt2img" }, "sampler")).toEqual({ tab: "txt2img", values: { s: 2 }, core: { a: 1 }, scripts: { seed: { seed: 3 } } })
    expect(actionRequest({ store, tab: "txt2img" }, "core", { selected: "seed.seed" })).toEqual({
      selected: "seed.seed",
      tab: "txt2img",
      values: { a: 1 },
      scripts: { sampler: { s: 2 }, seed: { seed: 3 } },
    })
  })
})

describe("conflict locks", () => {
  it("run for section enable toggles (on and off) with the whole form as context", async () => {
    const calls = mockServer({
      "/actions/core/__conflicts__": { "sampler.sampler_name": { enabled: false }, __toast__: { level: "warning", message: "locked" } },
      "/actions/core/hr_resolution": { __outputs__: { hr_final_resolution: "1024 → 2048" } },
    })
    const store = renderForm()
    act(() => store.getState().setValue("sampler", "sampler_name", "LCM"))
    fireEvent.click(screen.getByRole("switch", { name: "Enable Hires fix" }))
    await waitFor(() => expect(calls.map((c) => c.path)).toEqual(["/actions/core/hr_resolution", "/actions/core/__conflicts__"]))
    expect(calls[1]!.body).toEqual({
      selected: "core.enable_hr",
      tab: "txt2img",
      values: { enable_hr: true, hr_scale: 2 },
      scripts: { sampler: { sampler_name: "LCM" } },
    })
    // The patch locks the sampler choice; the output line is updated.
    await waitFor(() => expect(screen.getByRole("radiogroup", { name: "Sampler" })).toHaveAttribute("data-disabled"))
    expect(screen.getByText("1024 → 2048")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("switch", { name: "Enable Hires fix" }))
    await waitFor(() => expect(calls.filter((c) => c.path.endsWith("__conflicts__"))).toHaveLength(2))
  })

  it("run for choices of other owners", async () => {
    const calls = mockServer()
    renderForm()
    fireEvent.click(screen.getByRole("radio", { name: "LCM" }))
    await waitFor(() => expect(calls).toHaveLength(1))
    expect(calls[0]!.path).toBe("/actions/core/__conflicts__")
    expect(calls[0]!.body.selected).toBe("sampler.sampler_name")
    expect(calls[0]!.body.values).toEqual({ enable_hr: false, hr_scale: 2 })
    expect(calls[0]!.body.scripts).toEqual({ sampler: { sampler_name: "LCM" } })
  })

  it("re-derive locks after a draft restored a conflict param", async () => {
    const calls = mockServer()
    const store = createFormStore(`fx-${Math.random()}`, { persistDrafts: false })
    store.setState({ values: { sampler: { sampler_name: "LCM" } } })
    render(<TabForm schema={SCHEMA} formId="txt2img" store={store} />)
    await waitFor(() => expect(calls.map((c) => c.path)).toEqual(["/actions/core/__conflicts__"]))
    expect(calls[0]!.body).not.toHaveProperty("selected")
  })
})

describe("on_change", () => {
  it("is debounced for continuous kinds", async () => {
    const calls = mockServer()
    renderForm()
    fireEvent.click(screen.getByRole("button", { name: "Hires fix" }))
    const box = screen.getByLabelText("Upscale by", { selector: "input:not([type=range])" })
    fireEvent.change(box, { target: { value: "3" } })
    fireEvent.blur(box)
    fireEvent.change(box, { target: { value: "3.5" } })
    fireEvent.blur(box)
    expect(calls).toHaveLength(0)
    await waitFor(() => expect(calls).toHaveLength(1))
    expect(calls[0]!.path).toBe("/actions/core/hr_resolution")
    expect(calls[0]!.body.values).toMatchObject({ hr_scale: 3.5 })
  })

  it("bulk resets rerun each distinct on_change once, plus the conflict locks", async () => {
    const calls = mockServer()
    const store = renderForm()
    act(() => store.getState().setValues("core", { enable_hr: true, hr_scale: 3 }))
    fireEvent.click(screen.getByRole("button", { name: "Reset Hires fix to default" }))
    await waitFor(() => expect(calls).toHaveLength(2))
    expect(store.getState().values.core).toEqual({ enable_hr: false, hr_scale: 2 })
    expect(calls.map((c) => c.path)).toEqual(["/actions/core/hr_resolution", "/actions/core/__conflicts__"])
    expect(calls[1]!.body.selected).toBe("core.enable_hr")

    // Resetting several conflict params at once sends no `selected`.
    calls.length = 0
    act(() => {
      store.getState().setValue("core", "enable_hr", true)
      store.getState().setValue("sampler", "sampler_name", "LCM")
    })
    act(() => resetOwners({ store, tab: "txt2img" }, ["core", "sampler"]))
    await waitFor(() => expect(calls.filter((c) => c.path.endsWith("__conflicts__"))).toHaveLength(1))
    expect(calls.find((c) => c.path.endsWith("__conflicts__"))!.body.selected).toBeUndefined()
  })
})

describe("action responses", () => {
  it("take __settings__ into the cached options without writing them back", async () => {
    const calls = mockServer({ "/actions/core/__forge_preset__": { __settings__: { forge_preset: "xl" } } })
    queryClient.setQueryData(keys.app, { options: { forge_preset: "sd" } } as unknown as Bootstrap)
    const store = createFormStore("opt", { persistDrafts: false })
    await runAction({ store, tab: "txt2img" }, "core", "__forge_preset__", { preset: "xl" })
    expect(calls.map((c) => `${c.method} ${c.path}`)).toEqual(["POST /actions/core/__forge_preset__"])
    expect(calls[0]!.body.preset).toBe("xl")
    expect(queryClient.getQueryData<Bootstrap>(keys.app)?.options.forge_preset).toBe("xl")
  })

  it("drop a response overtaken by a newer call of the same action", async () => {
    const resolvers: ((value: Response) => void)[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(() => new Promise<Response>((resolve) => resolvers.push(resolve)))
    const store = createFormStore("stale", { persistDrafts: false })
    store.getState().load([{ owner: "core", layout: { type: "number", name: "n", label: "N" }, defaults: { n: 0 } }])
    const ctx = { store, tab: "txt2img" }
    const first = runAction(ctx, "core", "calc")
    const second = runAction(ctx, "core", "calc")
    await waitFor(() => expect(resolvers).toHaveLength(2))
    const respond = (n: number) => new Response(JSON.stringify({ n: { value: n } }), { headers: { "Content-Type": "application/json" } })
    resolvers[1]!(respond(2))
    expect(await second).not.toBeNull()
    resolvers[0]!(respond(1))
    expect(await first).toBeNull()
    expect(store.getState().values.core?.n).toBe(2)
  })
})
