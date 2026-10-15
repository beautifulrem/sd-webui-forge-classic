/** Display labels of the gallery (English keys, translated at render). */
import type { GalleryCategory } from "@/api/types"

export const CATEGORY_LABELS: Record<GalleryCategory, string> = {
  txt2img: "txt2img",
  img2img: "img2img",
  extras: "Extras",
  grids: "Grids",
  saved: "Saved",
  other: "Other",
}

export function formatBytes(size: number, locale?: string): string {
  const units = ["B", "KB", "MB", "GB"]
  let value = size
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit++
  }
  return `${value.toLocaleString(locale, { maximumFractionDigits: unit === 0 ? 0 : 1 })} ${units[unit]}`
}

/** Local date `YYYY-MM-DD`, `days` before today. */
export function daysAgo(days: number, now = new Date()): string {
  const date = new Date(now.getFullYear(), now.getMonth(), now.getDate() - days)
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`
}
