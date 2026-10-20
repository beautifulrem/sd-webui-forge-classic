/**
 * Where the toasts sit: bottom-right, lifted above whatever controls are
 * pinned to the bottom of the window under them (the phone's bottom
 * navigation and sticky Generate bar, the canvas's zoom and layer toolbars,
 * the settings' apply bar). Such controls carry `data-toast-obstacle`.
 */
import { useSonner } from "sonner"
import { useEffect, useState } from "react"

export interface Box {
  top: number
  bottom: number
  left: number
  right: number
}

/** sonner's own geometry (2.x): toast width, viewport offsets, phone breakpoint. */
export const TOAST_WIDTH = 356
export const OFFSET = 24
export const MOBILE_OFFSET = 16
export const MOBILE_MAX_WIDTH = 600
/** Room between a toast and the control under it (its close button sticks out ~7px). */
export const GAP = 12

export const OBSTACLE_ATTRIBUTE = "data-toast-obstacle"

/** The toaster's bottom offset (px) for a bottom-right toaster in a `width`×`height` window. */
export function toastBottomOffset(obstacles: readonly Box[], width: number, height: number): number {
  const mobile = width <= MOBILE_MAX_WIDTH
  const base = mobile ? MOBILE_OFFSET : OFFSET
  // the column the toasts occupy: the whole width on a phone, the right edge otherwise
  const left = mobile ? 0 : width - OFFSET - TOAST_WIDTH
  let offset = base
  for (const box of obstacles) {
    if (box.right - box.left <= 0 || box.bottom - box.top <= 0) continue
    if (box.right <= left || box.left >= width) continue
    // only controls along the bottom edge: anything else is covered for a moment at most
    if (box.top < height / 2 || box.top >= height) continue
    offset = Math.max(offset, height - box.top + GAP)
  }
  return Math.round(offset)
}

function measure(): number {
  const boxes = [...document.querySelectorAll<HTMLElement>(`[${OBSTACLE_ATTRIBUTE}]`)].map((element) => element.getBoundingClientRect())
  return toastBottomOffset(boxes, window.innerWidth, window.innerHeight)
}

/** The bottom offset to give the toaster; measured only while a toast is shown. */
export function useToastBottomOffset(): number {
  const { toasts } = useSonner()
  const active = toasts.length > 0
  const [offset, setOffset] = useState(OFFSET)

  useEffect(() => {
    if (!active) return
    let frame = 0
    const update = () => {
      if (frame) return
      frame = requestAnimationFrame(() => {
        frame = 0
        setOffset(measure())
      })
    }
    update() // in the next frame: the first toast is laid out then, before it slides in
    window.addEventListener("resize", update)
    window.addEventListener("scroll", update, { capture: true, passive: true })
    // obstacles come and go with the page (a route change, a collapsed pane)
    const mutations = new MutationObserver(update)
    const root = document.getElementById("root") ?? document.body
    mutations.observe(root, { childList: true, subtree: true })
    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener("resize", update)
      window.removeEventListener("scroll", update, { capture: true })
      mutations.disconnect()
    }
  }, [active])

  return offset
}
