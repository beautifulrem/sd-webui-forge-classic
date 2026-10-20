/**
 * Mock login (`--auth`). Off unless `sessionStorage["remi-mock-login"]` is
 * set (end-to-end tests set it before the page loads); then the only
 * account is `demo` / `demo`, and every other request is refused with 401
 * until it signed in, like the real server. The session lives in
 * `sessionStorage`, so it survives the reload after signing in.
 */
import { MockHttpError } from "./server"

type Handler = (match: RegExpExecArray, body: unknown, url: URL) => unknown | Promise<unknown>

const REQUIRED = "remi-mock-login"
const SESSION = "remi-mock-session"
export const MOCK_USER = "demo"
export const MOCK_PASSWORD = "demo"

const storage = () => (typeof sessionStorage === "undefined" ? null : sessionStorage)
const required = () => storage()?.getItem(REQUIRED) != null
const user = () => (required() ? (storage()?.getItem(SESSION) ?? null) : null)

/** The 401 answer for `path` (below /api/v2), or null when it may go on. */
export function mockRefusal(path: string): MockHttpError | null {
  if (!required() || user() !== null || path.startsWith("/auth/")) return null
  return new MockHttpError(401, { detail: "Not authenticated" })
}

export const authRoutes: [method: string, pattern: RegExp, handler: Handler][] = [
  ["GET", /^\/auth\/status$/, () => ({ required: required(), user: user() })],
  [
    "POST",
    /^\/auth\/login$/,
    (_m, body) => {
      const { username, password } = (body ?? {}) as { username?: string; password?: string }
      if (!required()) return undefined
      if (username !== MOCK_USER || password !== MOCK_PASSWORD) throw new MockHttpError(401, { detail: "wrong username or password" })
      storage()?.setItem(SESSION, username)
      return undefined
    },
  ],
  [
    "POST",
    /^\/auth\/logout$/,
    () => {
      storage()?.removeItem(SESSION)
      return undefined
    },
  ],
]
