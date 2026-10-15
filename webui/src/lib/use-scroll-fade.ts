/**
 * A horizontally scrolling strip (tab lists that do not fit their pane): its
 * edges fade where there is more to scroll to, and the selected item
 * (`[data-active]`) is kept in view when `active` changes.
 */
import { type CSSProperties, type RefObject, useEffect, useState } from "react"

export function useScrollFade(ref: RefObject<HTMLElement | null>, active: unknown): CSSProperties | undefined {
  const [edges, setEdges] = useState({ start: false, end: false })
  useEffect(() => {
    const strip = ref.current
    if (!strip) return
    const update = () => {
      const start = strip.scrollLeft > 1
      const end = strip.scrollLeft + strip.clientWidth < strip.scrollWidth - 1
      setEdges((old) => (old.start === start && old.end === end ? old : { start, end }))
    }
    update()
    const observer = new ResizeObserver(update)
    observer.observe(strip)
    strip.addEventListener("scroll", update, { passive: true })
    return () => {
      observer.disconnect()
      strip.removeEventListener("scroll", update)
    }
  }, [ref])
  useEffect(() => {
    const strip = ref.current
    const item = strip?.querySelector<HTMLElement>("[data-active]")
    if (!strip || !item) return
    const left = item.offsetLeft - strip.offsetLeft
    if (left < strip.scrollLeft) strip.scrollLeft = left - 16
    else if (left + item.offsetWidth > strip.scrollLeft + strip.clientWidth) strip.scrollLeft = left + item.offsetWidth - strip.clientWidth + 16
  }, [ref, active])
  if (!edges.start && !edges.end) return undefined
  const mask = `linear-gradient(to right, transparent, #000 ${edges.start ? "1.5rem" : "0rem"}, #000 calc(100% - ${edges.end ? "1.5rem" : "0rem"}), transparent)`
  return { maskImage: mask, WebkitMaskImage: mask }
}
