/**
 * The settings page's search text and filter mode. They live in a module
 * store, so they survive leaving the page, and are mirrored into the route
 * (`#/settings?q=&filter=`, replace navigation) so links, reloads and
 * back/forward carry them. A route change made elsewhere (a "Review" link,
 * an edited address) wins over the remembered values; the page's own writes
 * are recognised (`written`) and not read back.
 */
import { useEffect, useLayoutEffect } from "react"
import { create } from "zustand"
import { currentRoute, navigate } from "@/app/router"
import { FILTER_MODES, type FilterMode } from "./settings-model"

type UrlFilters = { q: string | null; filter: string | null }

type SettingsFiltersState = {
  query: string
  mode: FilterMode
  /** What the page last wrote into the route (or adopted from it). */
  written: UrlFilters
}

export const INITIAL_FILTERS: SettingsFiltersState = { query: "", mode: "all", written: { q: null, filter: null } }

export const useSettingsFilters = create<SettingsFiltersState>()(() => INITIAL_FILTERS)

export function setSettingsQuery(query: string) {
  useSettingsFilters.setState({ query })
}

export function setSettingsMode(mode: FilterMode) {
  useSettingsFilters.setState({ mode })
}

export function parseMode(value: string | null | undefined): FilterMode {
  return FILTER_MODES.includes(value as FilterMode) ? (value as FilterMode) : "all"
}

/** The route values for the current filters (`null` = absent). */
export function filtersToUrl(query: string, mode: FilterMode): UrlFilters {
  return { q: query.trim() ? query : null, filter: mode === "all" ? null : mode }
}

/** `params` with `q` / `filter` set to `values` (absent when null). */
export function withFilters(params: URLSearchParams | undefined, values: UrlFilters): URLSearchParams {
  const next = new URLSearchParams(params)
  for (const [key, value] of Object.entries(values)) {
    if (value === null) next.delete(key)
    else next.set(key, value)
  }
  return next
}

/**
 * The patch that adopts a route's filters: values the page did not write
 * itself replace the remembered ones; absent values keep them.
 */
export function adoptFromUrl(state: SettingsFiltersState, url: UrlFilters): Partial<SettingsFiltersState> | null {
  if (url.q === state.written.q && url.filter === state.written.filter) return null
  const patch: Partial<SettingsFiltersState> = { written: url }
  if (url.q !== null && url.q !== state.written.q) patch.query = url.q
  if (url.filter !== null && url.filter !== state.written.filter) patch.mode = parseMode(url.filter)
  return patch
}

const WRITE_DELAY = 250

/** Keeps the store and the route in step while the settings page is mounted. */
export function useSyncFiltersWithRoute(urlQuery: string | null, urlFilter: string | null) {
  const query = useSettingsFilters((s) => s.query)
  const mode = useSettingsFilters((s) => s.mode)

  useLayoutEffect(() => {
    const patch = adoptFromUrl(useSettingsFilters.getState(), { q: urlQuery, filter: urlFilter })
    if (patch) useSettingsFilters.setState(patch)
  }, [urlQuery, urlFilter])

  useEffect(() => {
    const values = filtersToUrl(query, mode)
    if (values.q === urlQuery && values.filter === urlFilter) return
    // Typing writes after a pause (each write re-renders the shell); a mode change at once.
    const timer = setTimeout(
      () => {
        const route = currentRoute()
        if (route.name !== "settings") return
        useSettingsFilters.setState({ written: values })
        const params = withFilters(route.query, values)
        params.delete("section")
        navigate({ name: "settings", query: params.size > 0 ? params : undefined }, { replace: true })
      },
      values.q === urlQuery ? 0 : WRITE_DELAY,
    )
    return () => clearTimeout(timer)
  }, [query, mode, urlQuery, urlFilter])
}
