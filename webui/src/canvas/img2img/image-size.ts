/**
 * The pixel size of an image param's file, for helpers that need it without
 * the editor (the "Resize by" result size): the file's metadata when known
 * (an upload's response, or a sent result's output metadata, see
 * `describeFile`), otherwise read once from the image. Sizes are shared by
 * every reader: whoever loads the image (this hook, or the stage editor
 * loading the background) records it, and all readers update.
 */
import { useEffect } from "react"
import { create } from "zustand"
import type { Size } from "../core/geometry"
import { useUploadInfo } from "@/ui/uploads"

const useSizes = create<{ byId: Record<string, Size> }>()(() => ({ byId: {} }))
/** Ids whose image is being read (one load per id, whoever asked). */
const loading = new Set<string>()

/** The size of file `id` (null until known). */
export function useImageSize(id: string | null): Size | null {
  const info = useUploadInfo(id)
  const url = info?.url ?? null
  const known = useSizes((s) => (id ? s.byId[id] : undefined))
  const infoWidth = info?.width ?? 0
  const infoHeight = info?.height ?? 0
  const described = infoWidth > 0 && infoHeight > 0

  useEffect(() => {
    if (!id || !url || described || loading.has(id) || useSizes.getState().byId[id]) return
    loading.add(id)
    const image = new Image()
    image.onload = () => {
      loading.delete(id)
      rememberImageSize(id, { width: image.naturalWidth, height: image.naturalHeight })
    }
    // A failed read is not cached: the next reader of this id tries again.
    image.onerror = () => loading.delete(id)
    image.src = url
  }, [id, url, described])

  if (!id) return null
  return described ? { width: infoWidth, height: infoHeight } : (known ?? null)
}

/** Records the size of file `id` (read elsewhere, e.g. by the editor); null forgets it (tests). */
export function rememberImageSize(id: string, size: Size | null) {
  if (size && !(size.width > 0 && size.height > 0)) return
  const current = useSizes.getState().byId[id]
  if (size && current?.width === size.width && current.height === size.height) return
  useSizes.setState((s) => {
    const byId = { ...s.byId }
    if (size) byId[id] = size
    else delete byId[id]
    return { byId }
  })
}
