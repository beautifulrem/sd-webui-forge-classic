/**
 * Pages put controls into the header with <HeaderSlot> (rendered through a
 * portal into the header's slot element, so pages keep ownership and state).
 */
import type { ReactNode } from "react"
import { createPortal } from "react-dom"
import { useHeaderSlotTarget } from "./header-slot-store"

export function HeaderSlot({ children }: { children: ReactNode }) {
  const element = useHeaderSlotTarget((s) => s.element)
  return element ? createPortal(children, element) : null
}
