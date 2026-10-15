/**
 * HTTP transport for /api/v2: the typed client generated from the server's
 * /openapi.json (`gen/schema.d.ts`, `pnpm gen:api`) and the error mapping.
 * Callers use the functions in `endpoints.ts`, never URLs.
 */
import createClient from "openapi-fetch"
import type { ValidationError } from "@/ui/schema"
import type { paths } from "./gen/schema"

declare global {
  interface Window {
    /** The server's root path under `--subpath` (index.html sets it; absent at the root). */
    __REMI_ROOT__?: string
  }
}

/** The app's mount path ("" at the server root, "/sd" under `--subpath /sd`). */
export const ROOT = (typeof window === "undefined" ? "" : (window.__REMI_ROOT__ ?? "")).replace(/\/+$/, "")

/** Path prefix of every API request (also the WebSocket's). */
export const API_BASE = `${ROOT}/api/v2`

/** Set at build time: `VITE_MOCK=1 pnpm dev` serves fixtures instead of the server. */
export const MOCK = import.meta.env.VITE_MOCK === "1"

export class ApiError extends Error {
  readonly status: number
  /** The decoded error body (see `validationErrors`). */
  readonly detail: unknown

  constructor(status: number, message: string, detail?: unknown) {
    super(message)
    this.name = "ApiError"
    this.status = status
    this.detail = detail
  }
}

export type Query = Record<string, string | number | boolean | null | undefined>

/** An API URL for `<img>`, `<a>` and `<audio>` (`path` is relative to /api/v2). */
export function apiUrl(path: string, query?: Query): string {
  let url = `${API_BASE}${path}`
  const qs = query ? new URLSearchParams(compactQuery(query)).toString() : ""
  if (qs) url += `?${qs}`
  return url
}

/** Drops unset query values (`undefined`, `null`, `""`) and stringifies the rest. */
function compactQuery(query: Query): Record<string, string> {
  const out: Record<string, string> = {}
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined && value !== null && value !== "") out[key] = String(value)
  }
  return out
}

async function transport(request: Request): Promise<Response> {
  if (MOCK) {
    const { handleMockRequest } = await import("./mock/server")
    return handleMockRequest(request)
  }
  const response = await fetch(request)
  // The session expired or the server now requires a login: the shell serves the login page.
  if (response.status === 401 && !request.url.includes("/auth/")) onUnauthorized()
  return response
}

let reloading = false
function onUnauthorized() {
  if (reloading || typeof location === "undefined") return
  reloading = true
  location.reload()
}

/** The generated, path-typed client. Paths are the server's (`/api/v2/...`); `ROOT` is the base URL. */
export const client = createClient<paths>({
  baseUrl: typeof location === "undefined" ? ROOT : `${location.origin}${ROOT}`,
  headers: { Accept: "application/json" },
  fetch: transport,
  querySerializer: (query) => new URLSearchParams(compactQuery((query ?? {}) as Query)).toString(),
})

type Result<T> = { data?: T; error?: unknown; response: Response }

/** The data of a generated-client call, or an `ApiError` for any non-2xx answer. */
export async function unwrap<T>(call: Promise<Result<T>>): Promise<T> {
  const { data, error, response } = await call
  if (!response.ok) throw new ApiError(response.status, errorMessage(response.status, error), error)
  return data as T
}

function isValidationError(value: unknown): value is ValidationError {
  if (!value || typeof value !== "object") return false
  const { owner, param, message } = value as Record<string, unknown>
  return typeof owner === "string" && typeof param === "string" && typeof message === "string"
}

/** The rejected values of a `{"errors": [{owner, param, message}]}` body (schema-format.md §3), or null. */
export function validationErrors(error: unknown): ValidationError[] | null {
  const payload = error instanceof ApiError ? error.detail : error
  if (!payload || typeof payload !== "object" || !("errors" in payload)) return null
  const errors = (payload as { errors: unknown }).errors
  if (!Array.isArray(errors)) return null
  const valid = errors.filter(isValidationError)
  return valid.length > 0 ? valid : null
}

/**
 * Something readable from an error body: Remi's validation `errors`, or
 * FastAPI's `{detail: string | [{msg, loc}]}`.
 */
function errorMessage(status: number, payload: unknown): string {
  const errors = validationErrors(payload)
  if (errors) return errors.map((e) => `${e.owner === "core" ? "" : `${e.owner}.`}${e.param}: ${e.message}`).join("; ")
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail: unknown }).detail
    if (typeof detail === "string") return detail
    if (Array.isArray(detail) && detail.length > 0) {
      return detail
        .map((d) => (d && typeof d === "object" && "msg" in d ? String((d as { msg: unknown }).msg) : String(d)))
        .join("; ")
    }
  }
  if (typeof payload === "string" && payload) return payload
  return `HTTP ${status}`
}
