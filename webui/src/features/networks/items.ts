/** Per-item helpers shared by the cards, the grid and the item menu (pure). */
import type { NetworkItem } from "@/api/types"

/** Where the item menu opens: an element, or the pointer (right click). */
export type MenuAnchor = HTMLElement | { x: number; y: number; element: HTMLElement }

/** The server's thumbnail widths (`remi.files.THUMB_WIDTHS`): asking for one of them keeps URLs stable while resizing. */
export const THUMB_WIDTHS = [64, 128, 192, 256, 320, 384, 512, 768, 1024] as const

/** The smallest thumbnail at least `pixels` wide. */
export const thumbWidth = (pixels: number): number => THUMB_WIDTHS.find((width) => width >= pixels) ?? 1024

/** Whether the item has a model file to keep metadata, a preview and a folder next to. */
export const isEditable = (item: Pick<NetworkItem, "editable">) => item.editable !== false
