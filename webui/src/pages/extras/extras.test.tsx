import { QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { extrasSchema } from "@/api/mock/fixtures"
import { useProgress } from "@/api/progress"
import { keys, queryClient } from "@/api/queries"
import { resetExtrasRuns, useExtrasRuns } from "@/features/extras/run"
import { getFormStore } from "@/ui/form-store"
import ExtrasPage from "./index"

function mockFetch() {
  const calls: { method: string; path: string; body: unknown }[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const request = input as Request
    const path = new URL(request.url).pathname.replace("/api/v2", "")
    const text = request.method === "GET" ? "" : await request.text()
    calls.push({ method: request.method, path, body: text ? JSON.parse(text) : undefined })
    const data = path === "/jobs" ? { job_id: "e1", job_ids: ["e1"], seq: 1 } : path === "/schema/extras" ? extrasSchema : {}
    return new Response(JSON.stringify(data), { status: path === "/jobs" ? 201 : 200, headers: { "Content-Type": "application/json" } })
  })
  return calls
}

describe("extras page", () => {
  beforeEach(() => {
    resetExtrasRuns()
    useProgress.setState({ running: null, owned: {}, last: {}, orphans: {} })
    queryClient.setQueryData(keys.schema("extras"), extrasSchema)
  })
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("renders the form from the schema and processes with Ctrl+Enter", async () => {
    const calls = mockFetch()
    render(
      <QueryClientProvider client={queryClient}>
        <ExtrasPage />
      </QueryClientProvider>,
    )
    expect(await screen.findByText("Scale by")).toBeInTheDocument()
    expect(screen.getByText("Results appear here, next to their source.")).toBeInTheDocument()
    act(() => getFormStore("extras").getState().setValue("core", "image", "src1")) // the source the schema requires
    fireEvent.keyDown(window, { key: "Enter", ctrlKey: true })
    await waitFor(() => expect(calls.some((call) => call.method === "POST" && call.path === "/jobs")).toBe(true))
    await waitFor(() => expect(screen.getByRole("button", { name: "Interrupt" })).toBeInTheDocument())
  })

  it("adds the form to the queue with Ctrl+Shift+Enter instead of processing it here", async () => {
    const calls = mockFetch()
    render(
      <QueryClientProvider client={queryClient}>
        <ExtrasPage />
      </QueryClientProvider>,
    )
    expect(await screen.findByText("Scale by")).toBeInTheDocument()
    act(() => getFormStore("extras").getState().setValue("core", "image", "src1"))
    fireEvent.keyDown(window, { key: "Enter", ctrlKey: true, shiftKey: true })
    await waitFor(() => expect(calls.filter((call) => call.method === "POST" && call.path === "/jobs")).toHaveLength(1))
    expect(calls.find((call) => call.path === "/jobs")?.body).toMatchObject({ tab: "extras" })
    expect(useExtrasRuns.getState().jobs).toEqual([])
    expect(screen.queryByRole("button", { name: "Interrupt" })).not.toBeInTheDocument()
  })

  it("compares a result with its source", async () => {
    mockFetch()
    useExtrasRuns.setState({ results: [{ jobId: "done", outputs: [{ id: "out1", kind: "image", width: 2048, height: 2048 }], source: "src1", info: "Postprocess upscale by: 4", at: 1 }] })
    render(
      <QueryClientProvider client={queryClient}>
        <ExtrasPage />
      </QueryClientProvider>,
    )
    expect(await screen.findByRole("slider", { name: "Comparison divider" })).toBeInTheDocument()
    expect(screen.getByRole("img", { name: "Source" })).toHaveAttribute("src", expect.stringContaining("/files/src1"))
    fireEvent.click(screen.getByRole("radio", { name: "Result" }))
    expect(screen.getByRole("img", { name: "Processed image" })).toHaveAttribute("src", expect.stringContaining("/files/out1"))
    expect(screen.getByText("Postprocess upscale by: 4")).toBeInTheDocument()
  })
})
