import type { FormTab } from "@/api/types"
import { type ResultItem, selectedItem } from "@/features/gallery/session-store"

/** The selected result when it is one of `tab`'s, else `tab`'s newest (the old button read the txt2img gallery). */
export function sourceItem(s: Parameters<typeof selectedItem>[0], tab: FormTab): ResultItem | null {
  const picked = selectedItem(s)
  return picked?.tab === tab ? picked : (s.items.find((item) => item.tab === tab) ?? null)
}
