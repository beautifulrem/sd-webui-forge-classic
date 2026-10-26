import { act, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it } from "vitest"
import { useSession } from "@/features/gallery/session-store"
import { HistoryPane } from "./HistoryPane"

describe("history pane", { timeout: 30_000 }, () => {
  beforeEach(() => useSession.getState().clear())

  it("shows the empty state without the grid, and the grid (loaded then) with the first result", async () => {
    render(<HistoryPane onOpen={() => {}} />)
    expect(screen.getByText("Results of this session appear here")).toBeInTheDocument()
    expect(screen.queryByRole("listbox")).toBeNull()
    act(() => useSession.getState().addJob({ jobId: "a", tab: "txt2img", outputs: [{ id: "f1", kind: "image" }] }))
    expect(await screen.findByRole("listbox", { name: "History" }, { timeout: 15_000 })).toBeInTheDocument()
    expect(screen.queryByText("Results of this session appear here")).toBeNull()
  })
})
