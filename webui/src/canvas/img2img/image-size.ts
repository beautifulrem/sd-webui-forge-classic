/**
 * The pixel size of an upload, for helpers that need it without the editor
 * (the "Resize by" result size): the upload's own metadata when the server
 * gave it, otherwise read once from the image and cached per id.
 */
import { useEffect, useState } from "react"
import type { Size } from "../core/geometry"
import { useUploadInfo } from "@/ui/uploads"

const sizes = new Map<string, Size>()

/** The size of upload `id` (null until known). */
export function useImageSize(id: string | null): Size | null {
  const info = useUploadInfo(id)
  const url = info?.url ?? null
  const [, setLoaded] = useState(0)
  const known = id ? sizes.get(id) : undefined
  const infoWidth = info?.width ?? 0
  const infoHeight = info?.height ?? 0
  const fromInfo = infoWidth > 0 && infoHeight > 0 ? { width: infoWidth, height: infoHeight } : null
  const described = fromInfo !== null

  useEffect(() => {
    if (!id || !url || sizes.has(id) || described) return
    let cancelled = false
    const image = new Image()
    image.onload = () => {
      sizes.set(id, { width: image.naturalWidth, height: image.naturalHeight })
      if (!cancelled) setLoaded((n) => n + 1)
    }
    image.src = url
    return () => {
      cancelled = true
    }
  }, [id, url, described])

  if (!id) return null
  return known ?? fromInfo
}

/** Tests: forget the cached sizes; or record one. */
export function rememberImageSize(id: string, size: Size | null) {
  if (size) sizes.set(id, size)
  else sizes.delete(id)
}
