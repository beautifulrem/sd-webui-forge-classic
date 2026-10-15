/** Gallery data: paged outputs per filter set, facets, and the actions on outputs. */
import { infiniteQueryOptions, queryOptions } from "@tanstack/react-query"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import type { GalleryItem, GalleryQuery } from "@/api/types"
import { navigate } from "@/app/router"
import { downloadFile } from "@/features/gallery/files"
import type { ResultItem } from "@/features/gallery/session-store"
import { t } from "@/i18n"
import { applyRemoved, applyStars, batches, galleryKeys, MAX_BATCH, PAGE_SIZE, refreshFacets, refreshGallery } from "./cache"
import { errorText } from "@/features/errors/error-text"

export { galleryKeys, MAX_BATCH, PAGE_SIZE }

export const galleryListQuery = (query: GalleryQuery) =>
  infiniteQueryOptions({
    queryKey: galleryKeys.list(query),
    queryFn: ({ pageParam, signal }) => api.gallery.list({ ...query, limit: PAGE_SIZE, ...(pageParam ? { cursor: pageParam } : {}) }, signal),
    initialPageParam: null as string | null,
    getNextPageParam: (page) => page.next_cursor ?? undefined,
    staleTime: 15_000,
  })

export const galleryFacetsQuery = queryOptions({ queryKey: galleryKeys.facets, queryFn: ({ signal }) => api.gallery.facets(signal), staleTime: 30_000 })

/** A gallery item as the viewer / result actions see it. */
export function asResult(item: GalleryItem): ResultItem {
  return {
    key: item.id,
    fileId: item.id,
    jobId: item.job_id ?? "",
    tab: item.tab === "img2img" || item.tab === "extras" ? item.tab : "txt2img",
    index: 0,
    infotext: item.infotext ?? null,
    width: item.width ?? undefined,
    height: item.height ?? undefined,
    createdAt: Date.parse(item.created_at) || 0,
  }
}

/** Stars or unstars outputs (in batches the server takes), updating the cached pages right away. */
export async function starOutputs(ids: readonly string[], starred: boolean) {
  applyStars(ids, starred)
  try {
    for (const batch of batches(ids)) await api.gallery.star(batch, starred)
    refreshFacets()
  } catch (error) {
    toast.error(t("Could not update the stars"), { description: errorText(error) })
    refreshGallery()
  }
}

/** Removes outputs from the gallery (and deletes the files with `deleteFiles`), in batches the server takes. */
export async function removeOutputs(ids: readonly string[], deleteFiles: boolean): Promise<boolean> {
  const id = toast.loading(deleteFiles ? t("Deleting…") : t("Removing…"))
  let count = 0
  const done: string[] = []
  try {
    for (const batch of batches(ids)) {
      count += (await api.gallery.remove(batch, deleteFiles)).count
      done.push(...batch)
    }
    toast.success(deleteFiles ? t("Deleted: {{count}}", { count }) : t("Removed from the gallery: {{count}}", { count }), { id })
    return true
  } catch (error) {
    toast.error(t("Could not remove the images"), { id, description: errorText(error) })
    return false
  } finally {
    applyRemoved(done)
    refreshFacets()
  }
}

/** Packs outputs into a zip and downloads it (at most `MAX_BATCH`: the first ones selected). */
export async function downloadZip(ids: readonly string[]) {
  const packed = ids.slice(0, MAX_BATCH)
  if (ids.length > MAX_BATCH) toast.warning(t("A zip holds at most {{count}} images: the first {{count}} selected are packed.", { count: MAX_BATCH }))
  const id = toast.loading(t("Packing {{count}} images…", { count: packed.length }))
  try {
    const result = await api.gallery.zip(packed)
    downloadFile(result.id)
    toast.success(t("Downloading {{name}}", { name: result.name }), { id })
  } catch (error) {
    toast.error(t("Could not pack the images"), { id, description: errorText(error, { 404: t("None of these images are on disk any more.") }) })
  }
}

/** Shows the file in the file manager of the WebUI machine. */
export async function revealOutput(fileId: string) {
  try {
    await api.gallery.reveal(fileId)
  } catch (error) {
    const forbidden = error instanceof ApiError && error.status === 403
    toast.error(forbidden ? t("Folders can only be opened on the WebUI machine") : t("Could not open the folder"), forbidden ? undefined : { description: errorText(error, { 404: t("The file is no longer on disk.") }) })
  }
}

/** Opens an output in PNG Info. */
export function openInPngInfo(fileId: string) {
  navigate({ name: "png-info", query: new URLSearchParams({ file: fileId }) })
}
