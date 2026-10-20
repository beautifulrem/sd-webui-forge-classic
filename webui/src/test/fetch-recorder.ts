/** Records the requests the API clients send (through `globalThis.fetch`) and answers them. */
import { vi } from "vitest"
import openapi from "@/api/openapi.json"

export type RecordedRequest = {
  method: string
  /** The URL path, still percent-encoded (`/api/v2/networks/lora/sub%2Fa/meta`). */
  path: string
  query: URLSearchParams
  /** The parsed JSON body, the `FormData` text of a multipart body, else `undefined`. */
  body: unknown
  contentType: string | null
}

type Answer = { status?: number; body?: unknown } | undefined

/**
 * Spies on `fetch`: every request is recorded (in order) and answered with `answer(request)`
 * (`{status, body}`; default 200 and `{}`). Restore with `vi.restoreAllMocks()`.
 */
export function recordFetch(answer: (request: RecordedRequest) => Answer = () => undefined) {
  const requests: RecordedRequest[] = []
  const spy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(input, init)
    const url = new URL(request.url)
    const contentType = request.headers.get("Content-Type")
    const text = request.body ? await request.clone().text() : ""
    const body = text && contentType?.includes("json") ? (JSON.parse(text) as unknown) : text || undefined
    const recorded = { method: request.method, path: url.pathname, query: url.searchParams, body, contentType }
    requests.push(recorded)
    const { status = 200, body: payload = {} } = answer(recorded) ?? {}
    return new Response(JSON.stringify(payload), { status, headers: { "Content-Type": "application/json" } })
  })
  return { requests, spy, last: () => requests.at(-1)! }
}

const ROUTES = Object.entries(openapi.paths as Record<string, Record<string, unknown>>).flatMap(([template, methods]) =>
  Object.keys(methods).map((method) => ({ method: method.toUpperCase(), pattern: new RegExp(`^${template.replace(/\{[^}]+\}/g, "[^/]+")}$`) })),
)

/** Whether the server's OpenAPI schema (`src/api/openapi.json`) declares `method` on this (encoded) path. */
export function declaredInOpenApi(method: string, path: string): boolean {
  return ROUTES.some((route) => route.method === method && route.pattern.test(path))
}
