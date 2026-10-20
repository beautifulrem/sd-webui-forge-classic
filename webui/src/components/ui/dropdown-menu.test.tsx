import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { Button } from "./button"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "./dropdown-menu"

function Menu({ onPick = () => {}, onOpenChange, onTriggerClick }: { onPick?: (item: string) => void; onOpenChange?: (open: boolean) => void; onTriggerClick?: () => void }) {
  return (
    <DropdownMenu onOpenChange={onOpenChange}>
      <DropdownMenuTrigger render={<Button variant="ghost" aria-label="More actions" onClick={onTriggerClick} />}>…</DropdownMenuTrigger>
      <DropdownMenuContent>
        <DropdownMenuItem onClick={() => onPick("download")}>Download</DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={() => onPick("reload")}>Reload</DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

// the first open transforms and loads the menu implementation: slow on a busy machine
const slow = { timeout: 15_000 }

describe("dropdown menu (implementation loaded on demand)", { timeout: 30_000 }, () => {
  it("renders the trigger as a menu button before the menu code is loaded", () => {
    render(<Menu />)
    const trigger = screen.getByRole("button", { name: "More actions" })
    expect(trigger).toHaveAttribute("aria-haspopup", "menu")
    expect(trigger).toHaveAttribute("aria-expanded", "false")
    expect(trigger).toHaveAttribute("data-slot", "dropdown-menu-trigger")
    expect(screen.queryByRole("menu")).toBeNull()
  })

  it("opens on press, runs an item and closes", async () => {
    const onPick = vi.fn()
    const onOpenChange = vi.fn()
    const onTriggerClick = vi.fn()
    render(<Menu onPick={onPick} onOpenChange={onOpenChange} onTriggerClick={onTriggerClick} />)
    const trigger = screen.getByRole("button", { name: "More actions" })
    fireEvent.click(trigger)
    const menu = await screen.findByRole("menu", {}, slow)
    // the consumer's own handler on the render element still runs
    expect(onTriggerClick).toHaveBeenCalledTimes(1)
    expect(trigger).toHaveAttribute("aria-expanded", "true")
    expect(trigger).toHaveAttribute("aria-controls", menu.id)
    expect(onOpenChange).toHaveBeenCalledWith(true, undefined)
    fireEvent.click(within(menu).getByRole("menuitem", { name: "Reload" }))
    expect(onPick).toHaveBeenCalledWith("reload")
    await waitFor(() => expect(trigger).toHaveAttribute("aria-expanded", "false"))
    expect(onOpenChange).toHaveBeenLastCalledWith(false, expect.anything())
  })

  it("opens from the keyboard with an arrow key and returns focus to the trigger on Escape", async () => {
    render(<Menu />)
    const trigger = screen.getByRole("button", { name: "More actions" })
    act(() => trigger.focus())
    fireEvent.keyDown(trigger, { key: "ArrowDown" })
    const menu = await screen.findByRole("menu", {}, slow)
    fireEvent.keyDown(menu, { key: "Escape" })
    await waitFor(() => expect(screen.queryByRole("menu")).toBeNull())
    await waitFor(() => expect(trigger).toHaveFocus())
  })

  it("closes again when the trigger is pressed while open", async () => {
    render(<Menu />)
    const trigger = screen.getByRole("button", { name: "More actions" })
    fireEvent.click(trigger)
    await screen.findByRole("menu", {}, slow)
    // a press on the trigger is an outside press for the popup, then a click on the trigger
    fireEvent.pointerDown(trigger)
    fireEvent.mouseDown(trigger)
    fireEvent.pointerUp(trigger)
    fireEvent.mouseUp(trigger)
    fireEvent.click(trigger)
    await waitFor(() => expect(trigger).toHaveAttribute("aria-expanded", "false"))
    await waitFor(() => expect(screen.queryByRole("menu")).toBeNull())
  })

  it("opens at once after an outside press elsewhere closed it", async () => {
    render(<Menu />)
    const trigger = screen.getByRole("button", { name: "More actions" })
    fireEvent.click(trigger)
    await screen.findByRole("menu", {}, slow)
    fireEvent.pointerDown(document.body, { clientX: 400, clientY: 300 })
    fireEvent.mouseDown(document.body, { clientX: 400, clientY: 300 })
    await waitFor(() => expect(trigger).toHaveAttribute("aria-expanded", "false"))
    fireEvent.click(trigger)
    expect(trigger).toHaveAttribute("aria-expanded", "true")
    expect(await screen.findByRole("menu", {}, slow)).toBeInTheDocument()
  })

  it("follows a controlled open state", async () => {
    const { rerender } = render(
      <DropdownMenu open={false}>
        <DropdownMenuTrigger>Open</DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuItem>One</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>,
    )
    expect(screen.queryByRole("menu")).toBeNull()
    rerender(
      <DropdownMenu open>
        <DropdownMenuTrigger>Open</DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuItem>One</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>,
    )
    expect(within(await screen.findByRole("menu", {}, slow)).getByRole("menuitem", { name: "One" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Open" })).toHaveAttribute("aria-expanded", "true")
  })
})
