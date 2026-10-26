import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { Button } from "./button"
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "./popover"

function Panel({ onOpenChange, onTriggerClick }: { onOpenChange?: (open: boolean) => void; onTriggerClick?: () => void }) {
  return (
    <Popover onOpenChange={onOpenChange}>
      <PopoverTrigger render={<Button variant="ghost" aria-label="Aspect ratio" onClick={onTriggerClick} />}>1:1</PopoverTrigger>
      <PopoverContent aria-labelledby="title">
        <PopoverTitle id="title">Aspect ratio</PopoverTitle>
        <button type="button">3:2</button>
      </PopoverContent>
    </Popover>
  )
}

const slow = { timeout: 15_000 }

describe("popover (implementation loaded on demand)", { timeout: 30_000 }, () => {
  it("renders the trigger as a dialog button before the popover code is loaded", () => {
    render(<Panel />)
    const trigger = screen.getByRole("button", { name: "Aspect ratio" })
    expect(trigger).toHaveAttribute("aria-haspopup", "dialog")
    expect(trigger).toHaveAttribute("aria-expanded", "false")
    expect(trigger).toHaveAttribute("data-slot", "popover-trigger")
    expect(screen.queryByRole("dialog")).toBeNull()
  })

  it("opens on press, named by its title, and returns the focus to the trigger on Escape", async () => {
    const onOpenChange = vi.fn()
    const onTriggerClick = vi.fn()
    render(<Panel onOpenChange={onOpenChange} onTriggerClick={onTriggerClick} />)
    const trigger = screen.getByRole("button", { name: "Aspect ratio" })
    act(() => trigger.focus())
    fireEvent.click(trigger)
    const dialog = await screen.findByRole("dialog", { name: "Aspect ratio" }, slow)
    expect(onTriggerClick).toHaveBeenCalledTimes(1)
    expect(onOpenChange).toHaveBeenCalledWith(true, undefined)
    expect(trigger).toHaveAttribute("aria-expanded", "true")
    expect(trigger).toHaveAttribute("aria-controls", dialog.id)
    fireEvent.keyDown(dialog, { key: "Escape" })
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
    expect(onOpenChange).toHaveBeenLastCalledWith(false, expect.anything())
    await waitFor(() => expect(trigger).toHaveFocus())
  })

  it("closes when the trigger is pressed while open", async () => {
    render(<Panel />)
    const trigger = screen.getByRole("button", { name: "Aspect ratio" })
    fireEvent.click(trigger)
    await screen.findByRole("dialog", {}, slow)
    // a press on the trigger is an outside press for the popup, then a click on the trigger
    fireEvent.pointerDown(trigger)
    fireEvent.mouseDown(trigger)
    fireEvent.pointerUp(trigger)
    fireEvent.mouseUp(trigger)
    fireEvent.click(trigger)
    await waitFor(() => expect(trigger).toHaveAttribute("aria-expanded", "false"))
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
  })

  it("follows a controlled open state", async () => {
    const { rerender } = render(
      <Popover open={false}>
        <PopoverTrigger>Open</PopoverTrigger>
        <PopoverContent aria-label="Settings">Body</PopoverContent>
      </Popover>,
    )
    expect(screen.queryByRole("dialog")).toBeNull()
    rerender(
      <Popover open>
        <PopoverTrigger>Open</PopoverTrigger>
        <PopoverContent aria-label="Settings">Body</PopoverContent>
      </Popover>,
    )
    expect(await screen.findByRole("dialog", { name: "Settings" }, slow)).toHaveTextContent("Body")
    expect(screen.getByRole("button", { name: "Open" })).toHaveAttribute("aria-expanded", "true")
  })
})
