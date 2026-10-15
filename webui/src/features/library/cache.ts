/**
 * Keeping the cached gallery pages current without refetching every loaded
 * page (an infinite query refetches all its pages, one after another).
 *
 * - New outputs (`job.completed`): newest-first lists merge the fresh first
 *   page in front; other sorts refetch when only one page is loaded, else
 *   are marked stale. Lists not on screen are dropped (they start again at
 *   their first page).
 * - Changes made here (star, remove) patch the pages in place; the server's
 *   `outputs.changed` echo of them is ignored.
 * - Changes made elsewhere: a few ids are fetched one by one and patched,
 *   more refresh the lists.
 *
 * Small on purpose: the WebSocket bridge (main bundle) imports it.
 */
import type { InfiniteData, Query } from "@tanstack/react-query"
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import { queryClient } from "@/api/queries"
import type { GalleryItem, GalleryQuery, Page } from "@/api/types"

export const PAGE_SIZE = 120
/** Ids the server takes per star / delete / zip request. */
export const MAX_BATCH = 2000
/** Remote changes patched item by item (more refresh the lists). */
const PATCH_LIMIT = 20
/** The server lists at most this many ids in `outputs.changed`. */
const EVENT_IDS = 200
const ECHO_MS = 10_000

export const galleryKeys = {
  all: ["gallery"] as const,
  lists: ["gallery", "list"] as const,
  list: (query: GalleryQuery) => ["gallery", "list", query] as const,
  facets: ["gallery", "facets"] as const,
}

type ListData = InfiniteData<Page<GalleryItem>, string | null>

function lists(): Query<ListData, Error, ListData, readonly unknown[]>[] {
  return queryClient.getQueryCache().findAll({ queryKey: galleryKeys.lists }) as Query<ListData, Error, ListData, readonly unknown[]>[]
}

const queryOf = (query: Query<ListData, Error, ListData, readonly unknown[]>) => (query.queryKey[2] ?? {}) as GalleryQuery

/** Splits ids into requests the server accepts. */
export function batches<T>(items: readonly T[], size = MAX_BATCH): T[][] {
  const out: T[][] = []
  for (let start = 0; start < items.length; start += size) out.push(items.slice(start, start + size))
  return out
}

/** Rewrites the items of every cached list (`edit` returns null to drop an item, which the total follows). */
function patch(edit: (item: GalleryItem, query: GalleryQuery) => GalleryItem | null) {
  for (const query of lists()) {
    const filters = queryOf(query)
    queryClient.setQueryData<ListData>(query.queryKey, (data) => {
      if (!data) return data
      let dropped = 0
      const pages = data.pages.map((page) => {
        const items: GalleryItem[] = []
        for (const item of page.items) {
          const next = edit(item, filters)
          if (next) items.push(next)
          else dropped++
        }
        return { ...page, items }
      })
      if (dropped && pages[0] && typeof pages[0].total === "number") pages[0] = { ...pages[0], total: Math.max(0, pages[0].total - dropped) }
      return { ...data, pages }
    })
  }
}

// Ids changed from this tab: the server's outputs.changed echo of them needs no refresh.
const recent = new Map<string, number>()
function remember(ids: readonly string[]) {
  const until = Date.now() + ECHO_MS
  for (const id of ids) recent.set(id, until)
  if (recent.size > 20_000) for (const [id, time] of recent) if (time < Date.now()) recent.delete(id)
}
const isEcho = (ids: readonly string[]) => ids.length > 0 && ids.every((id) => (recent.get(id) ?? 0) > Date.now())

/** Shows a star change in the cached pages (a starred-only list drops what was unstarred). */
export function applyStars(ids: readonly string[], starred: boolean) {
  remember(ids)
  const wanted = new Set(ids)
  patch((item, query) => (wanted.has(item.id) ? (query.starred && !starred ? null : { ...item, starred }) : item))
}

/** Drops removed outputs from the cached pages. */
export function applyRemoved(ids: readonly string[]) {
  remember(ids)
  const gone = new Set(ids)
  patch((item) => (gone.has(item.id) ? null : item))
}

export function refreshFacets() {
  void queryClient.invalidateQueries({ queryKey: galleryKeys.facets })
}

/** Refetches every gallery query (after a failed change: the cache may be wrong). */
export function refreshGallery() {
  void queryClient.invalidateQueries({ queryKey: galleryKeys.all })
}

/** New outputs were made: bring them into the lists without refetching every page. */
export async function onNewOutputs() {
  refreshFacets()
  await Promise.all(
    lists().map(async (query) => {
      const data = query.state.data
      if (!query.isActive() || !data) {
        queryClient.removeQueries({ queryKey: query.queryKey, exact: true })
        return
      }
      const filters = queryOf(query)
      if ((filters.sort ?? "newest") !== "newest") {
        await queryClient.invalidateQueries({ queryKey: query.queryKey, exact: true, refetchType: data.pages.length <= 1 ? "active" : "none" })
        return
      }
      let first: Page<GalleryItem>
      try {
        first = await api.gallery.list({ ...filters, limit: PAGE_SIZE })
      } catch {
        return // the next change or visit catches up
      }
      queryClient.setQueryData<ListData>(query.queryKey, (current) => {
        if (!current) return current
        const known = new Set(current.pages.flatMap((page) => page.items.map((item) => item.id)))
        const overlap = first.items.findIndex((item) => known.has(item.id))
        // Nothing in common: more new outputs than a page; start over from the fresh page.
        if (overlap === -1) return first.items.length ? { pages: [first], pageParams: [null] } : current
        const fresh = first.items.slice(0, overlap)
        const [head, ...rest] = current.pages
        if (!head) return current
        return { ...current, pages: [{ ...head, items: [...fresh, ...head.items], total: first.total ?? head.total }, ...rest] }
      })
    }),
  )
}

/** `outputs.changed` from the server: patch what another tab or client changed. */
export async function onOutputsChanged(ids: readonly string[] | undefined) {
  if (ids && isEcho(ids)) return
  refreshFacets()
  if (!ids || ids.length === 0 || ids.length >= EVENT_IDS) return refreshGallery()
  const cached = new Set(lists().flatMap((query) => query.state.data?.pages.flatMap((page) => page.items.map((item) => item.id)) ?? []))
  const shown = ids.filter((id) => cached.has(id))
  if (shown.length === 0) return
  if (shown.length > PATCH_LIMIT) return refreshGallery()
  const fresh = new Map<string, GalleryItem | null>()
  await Promise.all(
    shown.map(async (id) => {
      try {
        fresh.set(id, await api.gallery.get(id))
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) fresh.set(id, null)
      }
    }),
  )
  patch((item, query) => {
    if (!fresh.has(item.id)) return item
    const next = fresh.get(item.id) ?? null
    return next && !(query.starred && !next.starred) ? next : null
  })
}
