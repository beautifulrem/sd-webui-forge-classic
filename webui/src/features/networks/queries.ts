/**
 * Network browser queries (`/networks`); `networks.changed` invalidates
 * everything under `["networks"]`, and so does a change of the options the
 * server's answers depend on (see `NETWORK_OPTION_KEYS`).
 *
 * Lists are fetched as fixed windows (`offset` + `PAGE_SIZE`): the grid asks
 * for the pages covering what it shows, so jumping to the end of 5,000 items
 * loads one page rather than every page before it, and an invalidation only
 * refetches the pages on screen.
 */
import { queryOptions, useQueries, useQuery } from "@tanstack/react-query"
import { api } from "@/api/endpoints"
import type { NetworkItem, NetworkQuery } from "@/api/types"

/** Items per window: about two screens of cards. */
export const PAGE_SIZE = 100

export type SortKey = "name" | "date" | "created" | "size" | "path"
export type SortOrder = "asc" | "desc"

/** What the browser asks for (the window is added per page). */
export type ListParams = {
  q: string
  folder: string
  archs: readonly string[]
  compatible: boolean
  sort: SortKey
  order: SortOrder
}

/**
 * Options the server reads when it answers (the preset for "compatible" and
 * the mismatch badges, the LoRA prompt's weight and name, hidden folders).
 */
export const NETWORK_OPTION_KEYS: ReadonlySet<string> = new Set([
  "forge_preset",
  "extra_networks_default_multiplier",
  "lora_preferred_name",
  "extra_networks_hidden_models",
  "extra_networks_show_hidden_directories",
])

/** Whether an `options.changed` event (keys unknown: anything) makes the network lists stale. */
export const networksFollowOptions = (keys: readonly string[] | undefined) => !keys || keys.some((key) => NETWORK_OPTION_KEYS.has(key))

function toQuery(params: Partial<ListParams>): NetworkQuery {
  const query: NetworkQuery = {}
  if (params.q?.trim()) query.q = params.q.trim()
  if (params.folder) query.folder = params.folder
  if (params.archs?.length) query.arch = [...params.archs]
  if (params.compatible) query.compatible = true
  if (params.sort && params.sort !== "name") query.sort = params.sort
  if (params.order === "desc") query.order = "desc"
  return query
}

export const networkKeys = {
  all: ["networks"] as const,
  kinds: ["networks", "kinds"] as const,
  list: (kind: string, params: ListParams) => ["networks", "list", kind, params] as const,
  page: (kind: string, params: ListParams, page: number) => ["networks", "list", kind, params, page] as const,
  facets: (kind: string, params: Pick<ListParams, "q" | "folder" | "archs" | "compatible">) => ["networks", "facets", kind, params] as const,
  meta: (kind: string, name: string) => ["networks", "meta", kind, name] as const,
}

export const kindsQuery = queryOptions({
  queryKey: networkKeys.kinds,
  queryFn: ({ signal }) => api.networks.kinds(signal),
  staleTime: 5 * 60_000,
})

/** One window of a list (`page` × `PAGE_SIZE`). */
export const pageQuery = (kind: string, params: ListParams, page: number) =>
  queryOptions({
    queryKey: networkKeys.page(kind, params, page),
    queryFn: ({ signal }) => api.networks.list(kind, { ...toQuery(params), limit: PAGE_SIZE, offset: page * PAGE_SIZE }, signal),
    staleTime: 60_000,
  })

export const facetsQuery = (kind: string, params: Pick<ListParams, "q" | "folder" | "archs" | "compatible">) =>
  queryOptions({
    queryKey: networkKeys.facets(kind, params),
    queryFn: ({ signal }) => api.networks.facets(kind, toQuery(params), signal),
    // the previous counts stay while typing, but never another kind's
    placeholderData: (previous, previousQuery) => (previousQuery?.queryKey[2] === kind ? previous : undefined),
    staleTime: 60_000,
  })

export const metaQuery = (kind: string, name: string) =>
  queryOptions({ queryKey: networkKeys.meta(kind, name), queryFn: ({ signal }) => api.networks.meta(kind, name, signal), staleTime: 60_000 })

/** A stable key for an item (React keys, DOM ids). */
export const itemKey = (item: Pick<NetworkItem, "kind" | "name">) => `${item.kind}:${item.name}`

/** Item indices `[start, end]` (inclusive) the grid shows. */
export type IndexRange = { start: number; end: number }

/** The windows covering `range` and half a window around it (so scrolling on rarely waits). */
export function pagesFor(range: IndexRange, total: number): number[] {
  if (total <= 0 || range.end < range.start) return [0]
  const margin = PAGE_SIZE / 2
  const first = Math.floor(Math.max(0, range.start - margin) / PAGE_SIZE)
  const last = Math.floor(Math.min(total - 1, range.end + margin) / PAGE_SIZE)
  const pages: number[] = []
  for (let page = first; page <= last; page++) pages.push(page)
  return pages.length ? pages : [0]
}

export type NetworkWindow = {
  /** Items by index (`undefined`: not loaded yet); as long as `total`. */
  items: readonly (NetworkItem | undefined)[]
  total: number
  /** The first window has no data yet (a new kind; never for a new search of the same kind). */
  isPending: boolean
  isError: boolean
  error: Error | null
  isFetching: boolean
  /** The items shown are a previous search's while the new one loads. */
  isPlaceholder: boolean
  refetch: () => void
}

/**
 * The windows of a list covering `range`. The first window (which carries
 * the total) is always observed; a new search of the same kind keeps
 * showing the previous first window until it answers, a new kind does not.
 */
export function useNetworkWindow(kind: string, params: ListParams, range: IndexRange, enabled: boolean): NetworkWindow {
  const first = useQuery({
    ...pageQuery(kind, params, 0),
    enabled,
    placeholderData: (previous, previousQuery) => (previousQuery?.queryKey[2] === kind ? previous : undefined),
  })
  const total = first.data?.total ?? 0
  const more = first.isPlaceholderData ? [] : pagesFor(range, total).filter((page) => page > 0)
  const rest = useQueries({ queries: more.map((page) => ({ ...pageQuery(kind, params, page), enabled })) })

  const items: (NetworkItem | undefined)[] = Array.from({ length: Math.max(total, first.data?.items.length ?? 0) })
  const place = (page: number, data: { items: readonly NetworkItem[] } | undefined) => {
    data?.items.forEach((item, index) => {
      const at = page * PAGE_SIZE + index
      if (at < items.length) items[at] = item
    })
  }
  place(0, first.data)
  more.forEach((page, index) => place(page, rest[index]?.data))

  const failed = first.isError ? first : rest.find((query) => query.isError)
  return {
    items,
    total,
    isPending: first.isPending,
    isError: Boolean(failed),
    error: failed?.error ?? null,
    isFetching: first.isFetching || rest.some((query) => query.isFetching),
    isPlaceholder: first.isPlaceholderData,
    refetch: () => {
      void first.refetch()
      for (const query of rest) if (query.isError) void query.refetch()
    },
  }
}
