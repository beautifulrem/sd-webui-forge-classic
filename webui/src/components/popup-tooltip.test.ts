import { act, renderHook } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { usePopupTooltip } from "./popup-tooltip"

describe("popup trigger tooltip", () => {
  it("stays closed when focus returns to the trigger after the popup closes", () => {
    const { result } = renderHook(() => usePopupTooltip())
    act(() => result.current.tooltip.onOpenChange(true, { reason: "trigger-hover" }))
    expect(result.current.tooltip.open).toBe(true)
    act(() => result.current.onPopupOpenChange(true))
    expect(result.current.tooltip.open).toBe(false)
    // While the popup is open nothing opens the tooltip.
    act(() => result.current.tooltip.onOpenChange(true, { reason: "trigger-hover" }))
    expect(result.current.tooltip.open).toBe(false)
    act(() => result.current.onPopupOpenChange(false))
    // Focus comes back to the trigger: no tooltip (the Appearance menu bug).
    act(() => result.current.tooltip.onOpenChange(true, { reason: "trigger-focus" }))
    expect(result.current.tooltip.open).toBe(false)
    // Hovering still shows it.
    act(() => result.current.tooltip.onOpenChange(true, { reason: "trigger-hover" }))
    expect(result.current.tooltip.open).toBe(true)
  })

  it("focus shows it again once focus has moved on", () => {
    const { result } = renderHook(() => usePopupTooltip())
    act(() => result.current.onPopupOpenChange(true))
    act(() => result.current.onPopupOpenChange(false))
    act(() => result.current.trigger.onBlur())
    act(() => result.current.tooltip.onOpenChange(true, { reason: "trigger-focus" }))
    expect(result.current.tooltip.open).toBe(true)
  })
})
