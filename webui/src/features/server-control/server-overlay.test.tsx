/** The confirm dialog and reconnecting overlay of "Reload UI" / "Restart server". */
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import { dismissServerCommand, requestServerCommand, useServerControl } from "./server-control"
import ServerOverlay from "./ServerOverlay"
import { installBeforeUnloadGuard, useUnsaved } from "./unsaved"

function leaving(): boolean {
  const event = new Event("beforeunload", { cancelable: true })
  window.dispatchEvent(event)
  return event.defaultPrevented
}

afterEach(() => {
  act(() => dismissServerCommand())
  useUnsaved.getState().set("settings", 0)
  vi.restoreAllMocks()
})

describe("server overlay", () => {
  it("warns about unsaved edits, and a failed command keeps the close-tab prompt", async () => {
    installBeforeUnloadGuard()
    useUnsaved.getState().set("settings", 2)
    vi.spyOn(api.system, "ping").mockResolvedValue({ instance: "a" })
    vi.spyOn(api.system, "restart").mockRejectedValue(new ApiError(501, "the WebUI was not started by a launcher that can restart it"))
    render(<ServerOverlay />)
    act(() => requestServerCommand("restart"))
    expect(await screen.findByText("2 unsaved settings changes will be lost.")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Restart server" }))
    expect(await screen.findByText("This server cannot restart itself; restart it from its console.")).toBeInTheDocument()
    expect(useServerControl.getState().phase).toBe("failed")
    // Nothing reloaded the page: closing the tab still asks.
    expect(leaving()).toBe(true)
    fireEvent.click(screen.getByRole("button", { name: "Close" }))
    await waitFor(() => expect(useServerControl.getState().phase).toBe("idle"))
    expect(leaving()).toBe(true)
  })

  it("cancelling the confirmation runs nothing", async () => {
    const reload = vi.spyOn(api.system, "reload")
    render(<ServerOverlay />)
    act(() => requestServerCommand("reload"))
    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }))
    await waitFor(() => expect(useServerControl.getState().pending).toBeNull())
    expect(reload).not.toHaveBeenCalled()
  })
})
