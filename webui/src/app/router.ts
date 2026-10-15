/**
 * A tiny typed hash router. Hash routing keeps the server contract to "serve
 * index.html at /" — no catch-all needed next to /api, /files and /sdapi.
 *
 *   #/txt2img  #/img2img  #/extras  #/png-info  #/gallery  #/networks
 *   #/queue  #/settings  #/extensions  #/page/<id>
 *
 * An optional query (`#/settings?q=sampler`) is kept in `route.query`.
 */
import { useSyncExternalStore } from "react"

export const GENERATE_TABS = ["txt2img", "img2img"] as const
/** The tabs sharing the generate view (extras has its own page). */
export type GenerateRouteTab = (typeof GENERATE_TABS)[number]

export const STATIC_PAGES = [
  "extras",
  "png-info",
  "gallery",
  "networks",
  "queue",
  "settings",
  "extensions",
] as const
export type StaticPage = (typeof STATIC_PAGES)[number]

export type Route = (
  | { name: "generate"; tab: GenerateRouteTab }
  | { name: StaticPage }
  | { name: "page"; id: string }
  | { name: "not-found"; path: string }
) & { query?: URLSearchParams }

export const DEFAULT_ROUTE: Route = { name: "generate", tab: "txt2img" }

function isOneOf<T extends string>(list: readonly T[], value: string): value is T {
  return (list as readonly string[]).includes(value)
}

export function parseHash(hash: string): Route {
  const raw = hash.replace(/^#\/?/, "")
  const [pathPart = "", search] = raw.split("?", 2)
  let segments: string[]
  try {
    segments = pathPart.split("/").filter(Boolean).map(decodeURIComponent)
  } catch {
    // A malformed escape (`#/page/%E0%A4`) is a bad link, not a crash.
    return { name: "not-found", path: pathPart }
  }
  const query = search ? new URLSearchParams(search) : undefined
  const [head, ...rest] = segments
  if (!head) return DEFAULT_ROUTE
  if (isOneOf(GENERATE_TABS, head) && rest.length === 0) return { name: "generate", tab: head, query }
  if (isOneOf(STATIC_PAGES, head) && rest.length === 0) return { name: head, query }
  const [id] = rest
  if (head === "page" && id !== undefined && rest.length === 1) return { name: "page", id, query }
  return { name: "not-found", path: pathPart }
}

export function routeHref(route: Route): string {
  let path: string
  switch (route.name) {
    case "generate":
      path = route.tab
      break
    case "page":
      path = `page/${encodeURIComponent(route.id)}`
      break
    case "not-found":
      path = route.path
      break
    default:
      path = route.name
  }
  const query = route.query?.toString()
  return `#/${path}${query ? `?${query}` : ""}`
}

/** A stable string identity, e.g. for keys and active-state comparisons. */
export function routeKey(route: Route): string {
  return routeHref({ ...route, query: undefined })
}

export function navigate(route: Route, { replace = false } = {}) {
  const href = routeHref(route)
  if (replace) history.replaceState(null, "", href)
  else if (location.hash !== href) history.pushState(null, "", href)
  // pushState/replaceState do not fire hashchange.
  notify()
}

const listeners = new Set<() => void>()
let cachedHash: string | null = null
let cachedRoute: Route = DEFAULT_ROUTE

function notify() {
  for (const listener of listeners) listener()
}

function subscribe(listener: () => void) {
  listeners.add(listener)
  if (listeners.size === 1) {
    window.addEventListener("hashchange", notify)
    window.addEventListener("popstate", notify)
  }
  return () => {
    listeners.delete(listener)
    if (listeners.size === 0) {
      window.removeEventListener("hashchange", notify)
      window.removeEventListener("popstate", notify)
    }
  }
}

function snapshot(): Route {
  if (location.hash !== cachedHash) {
    cachedHash = location.hash
    cachedRoute = parseHash(location.hash)
  }
  return cachedRoute
}

export function useRoute(): Route {
  return useSyncExternalStore(subscribe, snapshot, () => DEFAULT_ROUTE)
}

export function currentRoute(): Route {
  return snapshot()
}
