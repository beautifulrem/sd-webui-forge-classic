import { api } from "@/api/endpoints"

export type ImageRef = { src: string; caption?: string }

const IMAGE_PROTOCOLS = new Set(["http:", "https:", "blob:"])

/** An image URL an output may name (it also becomes a link): http(s), blob,
 * raster data URLs, or same-site paths; null for anything else (`javascript:`...). */
export function safeImageUrl(url: string): string | null {
  const value = url.trim()
  if (/^data:image\/[a-z0-9.+-]+[;,]/i.test(value)) return value
  try {
    const parsed = new URL(value, "http://relative.invalid/")
    if (!IMAGE_PROTOCOLS.has(parsed.protocol)) return null
    return value.startsWith("/") || /^(https?|blob):/i.test(value) ? value : null
  } catch {
    return null
  }
}

/** Outputs hold file ids, URLs/data URLs, or `{id | url, caption?}` objects. */
export function toImageRef(item: unknown): ImageRef | null {
  if (typeof item === "string") {
    if (/^\s*([a-z][a-z0-9+.-]*:|\/)/i.test(item)) {
      const src = safeImageUrl(item)
      return src ? { src } : null
    }
    return item ? { src: api.fileUrl(item) } : null
  }
  if (item && typeof item === "object") {
    const record = item as { id?: unknown; url?: unknown; caption?: unknown; label?: unknown }
    const caption = typeof record.caption === "string" ? record.caption : typeof record.label === "string" ? record.label : undefined
    if (typeof record.url === "string") {
      const src = safeImageUrl(record.url)
      return src ? { src, caption } : null
    }
    if (typeof record.id === "string") return { src: api.fileUrl(record.id), caption }
  }
  return null
}
