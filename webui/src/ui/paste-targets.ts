/**
 * Where a pasted image goes when no image field is hovered or focused (those
 * take the paste themselves and `preventDefault` it): the first empty image
 * field on screen, else the last one — A1111's `dragdrop.js` rule. Image
 * inputs (`DropZone` with an image `accept`) and the image editor register
 * here while mounted.
 */
import { useEffect, useRef } from "react"

export type PasteTarget = {
  element: () => HTMLElement | null
  /** Whether the field holds no image yet. */
  empty: () => boolean
  deliver: (files: File[]) => void
}

const targets = new Set<PasteTarget>()

function visible(element: HTMLElement): boolean {
  // display: none / visibility: hidden here or above (a hidden tab, a closed panel)
  if (typeof element.checkVisibility === "function" && !element.checkVisibility({ visibilityProperty: true })) return false
  const rect = element.getBoundingClientRect()
  const height = window.innerHeight || document.documentElement.clientHeight
  const width = window.innerWidth || document.documentElement.clientWidth
  // jsdom has no layout (all zeros): count the element as on screen
  if (rect.width === 0 && rect.height === 0) return true
  return rect.bottom > 0 && rect.right > 0 && rect.top < height && rect.left < width
}

/** The target a pasted image goes to: the first empty one on screen (document order), else the last one on screen. */
export function pasteTarget(candidates: Iterable<PasteTarget> = targets): PasteTarget | null {
  const shown = [...candidates]
    .map((target) => ({ target, element: target.element() }))
    .filter((entry): entry is { target: PasteTarget; element: HTMLElement } => entry.element !== null && entry.element.isConnected && visible(entry.element))
    .sort((a, b) => (a.element.compareDocumentPosition(b.element) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1))
  return (shown.find((entry) => entry.target.empty()) ?? shown.at(-1))?.target ?? null
}

function imageFiles(event: ClipboardEvent): File[] {
  return Array.from(event.clipboardData?.files ?? []).filter((file) => file.type.startsWith("image/"))
}

let listening = false

function onPaste(event: ClipboardEvent) {
  // The clipboard is readable during the event only; who handles it is known once every listener ran.
  const files = imageFiles(event)
  if (files.length === 0) return
  setTimeout(() => {
    if (event.defaultPrevented) return // a hovered or focused field (or a page such as PNG Info) took it
    pasteTarget()?.deliver(files)
  }, 0)
}

export function registerPasteTarget(target: PasteTarget): () => void {
  targets.add(target)
  if (!listening) {
    window.addEventListener("paste", onPaste)
    listening = true
  }
  return () => {
    targets.delete(target)
    if (targets.size === 0 && listening) {
      window.removeEventListener("paste", onPaste)
      listening = false
    }
  }
}

/** Registers the field while mounted and `enabled`; the callbacks may change between renders. */
export function usePasteTarget(enabled: boolean, target: PasteTarget) {
  const latest = useRef(target)
  useEffect(() => {
    latest.current = target
  })
  useEffect(() => {
    if (!enabled) return
    return registerPasteTarget({
      element: () => latest.current.element(),
      empty: () => latest.current.empty(),
      deliver: (files) => latest.current.deliver(files),
    })
  }, [enabled])
}
