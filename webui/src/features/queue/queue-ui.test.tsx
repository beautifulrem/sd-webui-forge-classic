import { QueryClientProvider } from "@tanstack/react-query"
import { configure, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { keys, queryClient } from "@/api/queries"
import type { Job } from "@/api/types"
import { jobOf } from "@/test/api-fixtures"
import { getFormStore } from "@/ui/form-store"
import { tabOwners } from "@/ui/tab"
import { cancelEdit, enqueueForm, openJob, saveEdit, useQueueEdit } from "./edit-session"
import { HistoryList } from "./HistoryList"
import { ImportDialog } from "./ImportDialog"
import { PendingList } from "./PendingList"

type Call = { method: string; path: string; body: unknown }

// Rendering is slow in jsdom when the whole suite runs in parallel.
configure({ asyncUtilTimeout: 5_000 })

/** Answers `fetch` from `routes` ("METHOD /path" → [status, body]); records every call. */
function mockFetch(routes: Record<string, [number, unknown] | ((body: unknown) => [number, unknown])>) {
  const calls: Call[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const request = input as Request
    const path = new URL(request.url).pathname.replace("/api/v2", "")
    const text = request.method === "GET" ? "" : await request.text()
    const body = text ? (JSON.parse(text) as unknown) : undefined
    calls.push({ method: request.method, path, body })
    const route = routes[`${request.method} ${path}`]
    const [status, data] = typeof route === "function" ? route(body) : (route ?? [200, {}])
    return new Response(status === 204 ? null : JSON.stringify(data), { status, headers: { "Content-Type": "application/json" } })
  })
  return calls
}

const queued = (id: string, seq: number, priority = 0): Job => ({ ...jobOf(id, "queued"), seq, priority, summary: `job ${id}` })

function renderWithClient(ui: React.ReactNode) {
  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>)
}

afterEach(() => {
  vi.restoreAllMocks()
  queryClient.clear()
})

describe("pending list", { timeout: 20_000 }, () => {
  beforeEach(() => {
    // jsdom has no layout: give the scroller (and rows) a size so the virtualizer renders.
    vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockReturnValue(58)
    vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockReturnValue(600)
    vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockReturnValue(600)
  })

  it("reorders from the keyboard and sends the job's new neighbour", async () => {
    const calls = mockFetch({})
    const jobs = [queued("a", 1), queued("b", 2), queued("c", 3)]
    queryClient.setQueryData(keys.queueItems, jobs)
    renderWithClient(<PendingList jobs={jobs} />)
    const grip = screen.getByRole("button", { name: "Move job #3" })
    grip.focus()
    fireEvent.keyDown(grip, { key: " " })
    expect(screen.getByRole("status")).toHaveTextContent(/Picked up job #3 at position 3/)
    fireEvent.keyDown(screen.getByRole("button", { name: "Move job #3" }), { key: "ArrowUp" })
    fireEvent.keyDown(screen.getByRole("button", { name: "Move job #3" }), { key: "ArrowUp" })
    expect(screen.getAllByRole("listitem").map((item) => item.textContent)).toEqual([expect.stringContaining("job c"), expect.stringContaining("job a"), expect.stringContaining("job b")])
    fireEvent.keyDown(screen.getByRole("button", { name: "Move job #3" }), { key: "Enter" })
    await waitFor(() => expect(calls.find((call) => call.path === "/queue/move")?.body).toEqual({ id: "c", after: null, before: "a", priority: null }))
    expect(calls.some((call) => call.method === "PATCH" || call.path === "/queue/reorder")).toBe(false)
    expect(queryClient.getQueryData<Job[]>(keys.queueItems)?.map((job) => job.id)).toEqual(["c", "a", "b"])
  })

  it("changes the priority of a job moved into another priority group", async () => {
    const calls = mockFetch({})
    const jobs = [queued("urgent", 1, 10), queued("normal", 2)]
    renderWithClient(<PendingList jobs={jobs} />)
    const grip = screen.getByRole("button", { name: "Move job #2" })
    fireEvent.keyDown(grip, { key: "Enter" })
    fireEvent.keyDown(grip, { key: "ArrowUp" })
    fireEvent.keyDown(screen.getByRole("button", { name: "Move job #2" }), { key: " " })
    await waitFor(() => expect(calls.find((call) => call.path === "/queue/move")?.body).toEqual({ id: "normal", after: null, before: "urgent", priority: 10 }))
  })

  it("moves the dragged job, not its neighbour, when the head job starts mid-drag", async () => {
    const calls = mockFetch({})
    const jobs = [queued("a", 1), queued("b", 2), queued("c", 3), queued("d", 4)]
    const view = renderWithClient(<PendingList jobs={jobs} />)
    const grip = screen.getByRole("button", { name: "Move job #4" })
    fireEvent.keyDown(grip, { key: " " })
    fireEvent.keyDown(screen.getByRole("button", { name: "Move job #4" }), { key: "ArrowUp" })
    fireEvent.keyDown(screen.getByRole("button", { name: "Move job #4" }), { key: "ArrowUp" }) // a, d, b, c
    view.rerender(
      <QueryClientProvider client={queryClient}>
        <PendingList jobs={jobs.slice(1)} />
      </QueryClientProvider>,
    ) // "a" started: the queue is now b, c, d
    expect(screen.getAllByRole("listitem").map((item) => item.textContent)).toEqual([expect.stringContaining("job d"), expect.stringContaining("job b"), expect.stringContaining("job c")])
    fireEvent.keyDown(screen.getByRole("button", { name: "Move job #4" }), { key: " " })
    await waitFor(() => expect(calls.find((call) => call.path === "/queue/move")?.body).toEqual({ id: "d", after: null, before: "b", priority: null }))
  })

  it("does not move a job that started while it was held", async () => {
    const calls = mockFetch({})
    const jobs = [queued("a", 1), queued("b", 2)]
    const view = renderWithClient(<PendingList jobs={jobs} />)
    fireEvent.keyDown(screen.getByRole("button", { name: "Move job #1" }), { key: " " })
    fireEvent.keyDown(screen.getByRole("button", { name: "Move job #1" }), { key: "ArrowDown" })
    view.rerender(
      <QueryClientProvider client={queryClient}>
        <PendingList jobs={jobs.slice(1)} />
      </QueryClientProvider>,
    )
    fireEvent.keyDown(screen.getByRole("button", { name: "Move job #1" }), { key: " " })
    expect(screen.getByRole("status")).toHaveTextContent("The job started meanwhile and was not moved.")
    expect(calls).toEqual([])
  })

  it("renders only the rows near the viewport of a long queue and moves with one neighbour", async () => {
    const calls = mockFetch({})
    const jobs = Array.from({ length: 600 }, (_, index) => queued(`j${index}`, index + 1))
    renderWithClient(<PendingList jobs={jobs} />)
    const rows = screen.getAllByRole("listitem")
    expect(rows.length).toBeLessThan(60)
    expect(rows[0]).toHaveAttribute("aria-setsize", "600")
    fireEvent.keyDown(screen.getByRole("button", { name: "Move job #2" }), { key: " " })
    fireEvent.keyDown(screen.getByRole("button", { name: "Move job #2" }), { key: "ArrowUp" })
    fireEvent.keyDown(screen.getByRole("button", { name: "Move job #2" }), { key: " " })
    await waitFor(() => expect(calls.find((call) => call.path === "/queue/move")?.body).toEqual({ id: "j1", after: null, before: "j0", priority: null }))
  })

  it("cancels a keyboard move with Escape", () => {
    const calls = mockFetch({})
    const jobs = [queued("a", 1), queued("b", 2)]
    renderWithClient(<PendingList jobs={jobs} />)
    const grip = screen.getByRole("button", { name: "Move job #1" })
    fireEvent.keyDown(grip, { key: " " })
    fireEvent.keyDown(grip, { key: "ArrowDown" })
    fireEvent.keyDown(screen.getByRole("button", { name: "Move job #1" }), { key: "Escape" })
    expect(screen.getAllByRole("listitem")[0]).toHaveTextContent("job a")
    expect(screen.getByRole("status")).toHaveTextContent("Move cancelled")
    expect(calls).toEqual([])
  })
})

describe("history list", () => {
  beforeEach(() => {
    vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockReturnValue(600)
    vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockReturnValue(800)
  })

  it("shows status, duration and actions; imported jobs that cannot run say so", async () => {
    const calls = mockFetch({})
    const done: Job = { ...jobOf("done", "completed", [{ id: "out1", kind: "image" }]), seq: 4, summary: "a fox", started_at: "2026-01-01T00:00:00Z", finished_at: "2026-01-01T00:00:12Z" }
    const imported: Job = { ...jobOf("old", "cancelled"), seq: 5, summary: "imported", replayable: false, source: "agent-scheduler", notes: [{ message: "Script arguments past the current parameters could not be mapped: {{count}}.", values: { count: 2 } }] }
    const onView = vi.fn()
    renderWithClient(<HistoryList jobs={[done, imported]} total={2} hasMore={false} loadingMore={false} onLoadMore={() => {}} onView={onView} />)
    const rows = screen.getAllByRole("listitem")
    expect(rows[0]).toHaveTextContent("Completed")
    expect(rows[0]).toHaveTextContent("12 s")
    fireEvent.click(screen.getByRole("button", { name: "View image 1 of job #4" }))
    expect(onView).toHaveBeenCalledWith(done, 0)

    const queueAgain = screen.getAllByRole("button", { name: "Queue again" })
    expect(queueAgain[0]).toBeEnabled()
    expect(queueAgain[1]).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Not replayable as it is: open it in its form to queue it" }))
    expect(screen.getByText(/could not be mapped: 2\./)).toBeInTheDocument()

    fireEvent.click(screen.getAllByRole("button", { name: "Star" })[0]!)
    await waitFor(() => expect(calls.find((call) => call.path === "/jobs/star")?.body).toEqual({ ids: ["done"], starred: true }))
  })

  it("shows a placeholder for an output deleted since, not a broken image", () => {
    mockFetch({})
    const done: Job = { ...jobOf("done", "completed", [{ id: "gone1", kind: "image" }]), seq: 4 }
    renderWithClient(<HistoryList jobs={[done]} total={1} hasMore={false} loadingMore={false} onLoadMore={() => {}} onView={() => {}} />)
    const thumb = screen.getByRole("button", { name: "View image 1 of job #4" })
    fireEvent.error(thumb.querySelector("img")!)
    expect(thumb.querySelector("img")).toBeNull()
    expect(thumb.querySelector("[title='The image is no longer available']")).not.toBeNull()
  })

  it("does not request the thumbnail of an output the server marks missing", () => {
    mockFetch({})
    const done: Job = { ...jobOf("done", "completed", [{ id: "gone2", kind: "image", missing: true }]), seq: 4 }
    renderWithClient(<HistoryList jobs={[done]} total={1} hasMore={false} loadingMore={false} onLoadMore={() => {}} onView={() => {}} />)
    const thumb = screen.getByRole("button", { name: "View image 1 of job #4" })
    expect(thumb.querySelector("img")).toBeNull()
    expect(thumb.querySelector("[title='The image is no longer available']")).not.toBeNull()
  })
})

describe("import dialog", { timeout: 20_000 }, () => {
  const report = { source: "agent-scheduler", path: null, total: 1, queued: 1, history: 0, refused: 0, invalid: 0, skipped: 0, errors: 0, paused: true, items: [] }

  it("keeps an import that finishes after the dialog was closed, with its resume prompt", async () => {
    let finish: (() => void) | undefined
    const done = new Promise<void>((resolve) => (finish = resolve))
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const request = input as Request
      const path = new URL(request.url).pathname.replace("/api/v2", "")
      if (path === "/queue/import") return Response.json([{ id: "agent-scheduler", label: "Agent Scheduler", available: true, count: 3, path: null }])
      if (path === "/queue/import/agent-scheduler") {
        await done
        return Response.json(report)
      }
      return Response.json({})
    })
    const view = renderWithClient(<ImportDialog open onOpenChange={() => {}} />)
    expect(await screen.findByText("Records: 3")).toBeInTheDocument() // no path under --hide-ui-dir-config
    fireEvent.click(screen.getByRole("button", { name: "Import" }))
    await screen.findByText(/You can close this dialog/)
    view.rerender(
      <QueryClientProvider client={queryClient}>
        <ImportDialog open={false} onOpenChange={() => {}} />
      </QueryClientProvider>,
    )
    finish?.()
    await waitFor(() => expect(vi.mocked(globalThis.fetch).mock.calls.length).toBeGreaterThanOrEqual(2))
    view.rerender(
      <QueryClientProvider client={queryClient}>
        <ImportDialog open onOpenChange={() => {}} />
      </QueryClientProvider>,
    )
    expect(await screen.findByRole("button", { name: "Resume the queue" })).toBeInTheDocument()
    expect(screen.getByText("The queue was paused so you can review the imported jobs first.")).toBeInTheDocument()
  })

  it("explains a missing database in the UI language's words, not the server's", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const path = new URL((input as Request).url).pathname.replace("/api/v2", "")
      if (path === "/queue/import") return Response.json([{ id: "agent-scheduler", label: "Agent Scheduler", available: true, count: 1, path: "/data/task_scheduler.sqlite3" }])
      return Response.json({ detail: "no Agent Scheduler database was found" }, { status: 404 })
    })
    renderWithClient(<ImportDialog open onOpenChange={() => {}} />)
    fireEvent.click(await screen.findByRole("button", { name: "Import" }))
    expect(await screen.findByRole("alert")).toHaveTextContent("Nothing to import was found on this machine.")
    expect(screen.queryByText(/no Agent Scheduler database/)).not.toBeInTheDocument()
  })
})

describe("editing a queued job in its form", () => {
  const schema = tabSchema("txt2img")
  beforeEach(() => {
    cancelEdit("txt2img")
    queryClient.setQueryData(keys.schema("txt2img"), schema)
    getFormStore("txt2img").getState().load(tabOwners(schema), { conflictParams: schema.conflict_params })
  })

  const detail = { ...queued("j1", 12), core: { prompt: "queued prompt" }, scripts: { seed: { seed: 42 } }, layout_hash: "", schema_version: 1 }

  it("loads the parameters, then updates the queued job with the form", async () => {
    const calls = mockFetch({ "GET /jobs/j1": [200, detail], "PUT /jobs/j1/params": [200, detail] })
    await openJob("j1")
    expect(useQueueEdit.getState().sessions.txt2img).toMatchObject({ jobId: "j1", seq: 12 })
    const store = getFormStore("txt2img")
    expect(store.getState().values.core?.prompt).toBe("queued prompt")
    expect(store.getState().values.seed?.seed).toBe(42)

    store.getState().setValue("core", "prompt", "edited prompt")
    expect(await saveEdit("txt2img")).toBe(true)
    const put = calls.find((call) => call.method === "PUT")!
    expect((put.body as { core: { prompt: string } }).core.prompt).toBe("edited prompt")
    expect(useQueueEdit.getState().sessions.txt2img).toBeUndefined()
  })

  it("marks rejected values and keeps editing; a started job ends the edit", async () => {
    mockFetch({
      "GET /jobs/j1": [200, detail],
      "PUT /jobs/j1/params": [422, { errors: [{ owner: "core", param: "prompt", message: "too long" }] }],
    })
    await openJob("j1")
    expect(await saveEdit("txt2img")).toBe(false)
    expect(getFormStore("txt2img").getState().errors.core?.prompt).toBe("too long")
    expect(useQueueEdit.getState().sessions.txt2img).toBeDefined()

    vi.restoreAllMocks()
    mockFetch({ "PUT /jobs/j1/params": [409, { detail: "only queued jobs can be edited" }] })
    expect(await saveEdit("txt2img")).toBe(false)
    expect(useQueueEdit.getState().sessions.txt2img).toBeUndefined()
  })

  it("opens a finished job without starting an edit", async () => {
    mockFetch({ "GET /jobs/j2": [200, { ...detail, id: "j2", status: "completed" }] })
    await openJob("j2")
    expect(useQueueEdit.getState().sessions.txt2img).toBeUndefined()
    expect(getFormStore("txt2img").getState().values.core?.prompt).toBe("queued prompt")
  })

  it("adds the form to the queue without tracking it as the form's run", async () => {
    const calls = mockFetch({ "POST /jobs": [201, { job_id: "new", job_ids: ["new"], seq: 9 }] })
    expect(await enqueueForm("txt2img")).toBe("new")
    const queued = calls.find((call) => call.path === "/jobs")
    expect(queued?.body).toMatchObject({ tab: "txt2img" })
  })
})

describe("queued count", () => {
  it("is 0 where it is disabled, even with the queue status cached", async () => {
    const { renderHook } = await import("@testing-library/react")
    const { useQueuedCount } = await import("./status")
    const { queueKeys } = await import("./queries")
    queryClient.setQueryData(queueKeys.status, { size: 7, paused: false, running: null })
    const wrapper = ({ children }: { children: React.ReactNode }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    expect(renderHook(() => useQueuedCount(false), { wrapper }).result.current).toBe(0)
    expect(renderHook(() => useQueuedCount(true), { wrapper }).result.current).toBe(7)
    queryClient.removeQueries({ queryKey: queueKeys.status })
  })
})

describe("loading states", () => {
  it("the running card is a placeholder until the queue state arrives, not 'Nothing is running'", async () => {
    const { RunningCard } = await import("./RunningCard")
    const { rerender } = render(
      <QueryClientProvider client={queryClient}>
        <RunningCard paused={false} queued={0} loading />
      </QueryClientProvider>,
    )
    expect(screen.queryByText("Nothing is running")).toBeNull()
    expect(screen.getByRole("region", { name: "Now running" })).toHaveAttribute("aria-busy", "true")
    rerender(
      <QueryClientProvider client={queryClient}>
        <RunningCard paused={false} queued={0} />
      </QueryClientProvider>,
    )
    expect(screen.getByText("Nothing is running")).toBeVisible()
  })
})
