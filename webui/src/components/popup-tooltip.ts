/**
 * Tooltip state for the trigger of a menu or popover. Closing a popup
 * returns focus to its trigger, which would open the trigger's tooltip and
 * leave it showing; this keeps the tooltip closed while the popup is open and
 * after it closes: focus coming back no longer opens it until focus moves
 * on (hovering still does).
 *
 *   const tip = usePopupTooltip()
 *   <DropdownMenu open={tip.popupOpen} onOpenChange={tip.onPopupOpenChange}>
 *     <Tooltip {...tip.tooltip}>
 *       <TooltipTrigger render={<DropdownMenuTrigger render={<Button {...tip.trigger} />} />} />
 */
import { useRef, useState } from "react"

export function usePopupTooltip() {
  const [tooltipOpen, setTooltipOpen] = useState(false)
  const [popupOpen, setPopupOpen] = useState(false)
  const suppressed = useRef(false)
  const release = () => {
    suppressed.current = false
  }
  return {
    popupOpen,
    onPopupOpenChange: (open: boolean) => {
      setPopupOpen(open)
      setTooltipOpen(false)
      if (!open) suppressed.current = true
    },
    tooltip: {
      open: tooltipOpen && !popupOpen,
      onOpenChange: (open: boolean, details?: { reason?: string }) => {
        if (open && popupOpen) return
        if (open && suppressed.current && details?.reason !== "trigger-hover") return
        if (open) release()
        setTooltipOpen(open)
      },
    },
    trigger: { onBlur: release },
  }
}
