/**
 * Low-level HTTP transport for /api/v2. Everything above this file deals in
 * typed functions (see `endpoints.ts`), so an OpenAPI-generated client can
 * replace those functions without touching callers.
 */

import type { ValidationError } from "@/ui/schema"

export const API_BASE = "/api/v2"

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

export type RequestOptions = {
  query?: Query
  /** JSON body (serialised) or FormData (sent as multipart). */
  body?: unknown
  signal?: AbortSignal
}

export function apiUrl(path: string, query?: Query): string {
  let url = `${API_BASE}${path}`
  if (query) {
    const params = new URLSearchParams()
    for (const [key, value] of Object.entries(query)) {
      if (value !== undefined && value !== null && value !== "") params.set(key, String(value))
    }
    const qs = params.toString()
    if (qs) url += `?${qs}`
  }
  return url
}

async function transport(request: Request): Promise<Response> {
  if (MOCK) {
    const { handleMockRequest } = await import("./mock/server")
    return handleMockRequest(request)
  }
  return fetch(request)
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

export async function request<T>(method: string, path: string, options: RequestOptions = {}): Promise<T> {
  const { query, body, signal } = options
  const headers = new Headers({ Accept: "application/json" })
  let payload: BodyInit | undefined
  if (body instanceof FormData) {
    payload = body
  } else if (body !== undefined) {
    headers.set("Content-Type", "application/json")
    payload = JSON.stringify(body)
  }

  const url = new URL(apiUrl(path, query), location.href)
  const response = await transport(new Request(url, { method, headers, body: payload, signal }))
  const isJson = response.headers.get("Content-Type")?.includes("application/json")
  const data: unknown = response.status === 204 ? undefined : isJson ? await response.json() : await response.text()
  if (!response.ok) throw new ApiError(response.status, errorMessage(response.status, data), data)
  return data as T
}

export const http = {
  get: <T>(path: string, options?: Omit<RequestOptions, "body">) => request<T>("GET", path, options),
  post: <T>(path: string, body?: unknown, options?: Omit<RequestOptions, "body">) =>
    request<T>("POST", path, { ...options, body }),
  patch: <T>(path: string, body?: unknown, options?: Omit<RequestOptions, "body">) =>
    request<T>("PATCH", path, { ...options, body }),
  delete: <T>(path: string, options?: Omit<RequestOptions, "body">) => request<T>("DELETE", path, options),
}
