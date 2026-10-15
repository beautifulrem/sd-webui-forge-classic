import { api } from "@/api/endpoints"

export type ImageRef = { src: string; caption?: string }

/** Outputs hold file ids, URLs/data URLs, or `{id | url, caption?}` objects. */
export function toImageRef(item: unknown): ImageRef | null {
  if (typeof item === "string") {
    if (/^(https?:|data:image\/|blob:|\/)/.test(item)) return { src: item }
    return item ? { src: api.fileUrl(item) } : null
  }
  if (item && typeof item === "object") {
    const record = item as { id?: unknown; url?: unknown; caption?: unknown; label?: unknown }
    const caption = typeof record.caption === "string" ? record.caption : typeof record.label === "string" ? record.label : undefined
    if (typeof record.url === "string") return { src: record.url, caption }
    if (typeof record.id === "string") return { src: api.fileUrl(record.id), caption }
  }
  return null
}
