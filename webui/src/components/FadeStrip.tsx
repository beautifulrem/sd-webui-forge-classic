/**
 * A row that scrolls sideways when it does not fit (filter chips on a
 * phone): its edges fade where there is more to scroll to, and the item
 * marked `data-active` is kept in view when `active` changes.
 */
import { type ComponentProps, useRef } from "react"
import { useScrollFade } from "@/lib/use-scroll-fade"
import { cn } from "@/lib/utils"

export function FadeStrip({ active, className, style, ...props }: ComponentProps<"div"> & { active?: unknown }) {
  const strip = useRef<HTMLDivElement>(null)
  const fade = useScrollFade(strip, active)
  return <div ref={strip} style={{ ...style, ...fade }} className={cn("overflow-x-auto [scrollbar-width:none]", className)} {...props} />
}
