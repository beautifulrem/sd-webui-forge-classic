/** Test helpers of the network browser: the mock `/networks` API over `fetch`, jsdom layout, rendering. */
import { QueryClientProvider } from "@tanstack/react-query"
import { render } from "@testing-library/react"
import { vi } from "vitest"
import { networkRoutes } from "@/api/mock/networks"
import { keys, queryClient } from "@/api/queries"
import type { Bootstrap } from "@/api/types"
import { bootstrapWith } from "@/test/api-fixtures"
import { NetworkBrowser } from "./NetworkBrowser"
import type { Surface } from "./store"

export type Call = { method: string; path: string; url: URL; body: unknown }

export const BASE_OPTIONS: Record<string, unknown> = { extra_networks_add_text_separator: " ", sd_model_checkpoint: null }

/** A bootstrap for the browser (`folders.open` on unless `features` says otherwise). */
export const networksBootstrap = (options: Record<string, unknown> = {}, features: string[] = ["folders.open"]): Bootstrap => ({
  ...bootstrapWith({ ...BASE_OPTIONS, ...options }),
  features,
})

type MockOptions = {
  options?: Record<string, unknown>
  features?: string[]
  /** Answer 500 for requests this returns true for. */
  fail?: (call: Call) => boolean
}

/** Serves the mock `/networks` routes, `/app`, `PATCH /options` and `POST /uploads` through `fetch`; returns the calls made. */
export function mockApi({ options: initial = {}, features, fail }: MockOptions = {}) {
  const calls: Call[] = []
  const options: Record<string, unknown> = { ...BASE_OPTIONS, ...initial }
  const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const request = input as Request
    const url = new URL(request.url)
    const path = url.pathname.replace("/api/v2", "")
    const isJson = request.headers.get("content-type")?.includes("application/json")
    const body = request.method === "GET" || !isJson ? undefined : await request.json().catch(() => undefined)
    const call = { method: request.method, path, url, body }
    calls.push(call)
    if (fail?.(call)) return json({ detail: "the server had a problem" }, 500)
    for (const [method, pattern, handler] of networkRoutes) {
      const match = method === request.method ? pattern.exec(path) : null
      if (match) return json(handler(match, body, url))
    }
    if (path === "/options" && request.method === "PATCH") Object.assign(options, body)
    if (path === "/uploads" && request.method === "POST") return json({ id: "upload-1", kind: "image", name: "preview.png" })
    return json(path === "/app" ? networksBootstrap(options, features) : { ok: true })
  })
  return calls
}

/** jsdom has no layout: give every element a size so the virtualizer renders rows. */
export function layout(width: number, height: number) {
  for (const prop of ["clientWidth", "offsetWidth"] as const) vi.spyOn(HTMLElement.prototype, prop, "get").mockReturnValue(width)
  for (const prop of ["clientHeight", "offsetHeight"] as const) vi.spyOn(HTMLElement.prototype, prop, "get").mockReturnValue(height)
}

export function renderBrowser({ form = "txt2img", surface = "page" }: { form?: string; surface?: Surface } = {}) {
  return render(
    <QueryClientProvider client={queryClient}>
      <NetworkBrowser surface={surface} form={form} />
    </QueryClientProvider>,
  )
}

/** Seeds the bootstrap the browser reads its settings from. */
export function seedBootstrap(options: Record<string, unknown> = {}, features?: string[]) {
  queryClient.setQueryData(keys.app, networksBootstrap(options, features))
}

/** Generous: the suites run in parallel with every other test file. */
export const slow = { timeout: 15_000 }

/** GET calls of a kind's list, with their query. */
export const listCalls = (calls: readonly Call[], kind = "lora") => calls.filter((call) => call.method === "GET" && call.path === `/networks/${kind}`)
