/** The queue controls next to a form's Generate button: "Add to queue" and the queued-job edit banner. */
import { QueryClientProvider } from "@tanstack/react-query"
import { configure, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { toast } from "sonner"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { keys, queryClient } from "@/api/queries"
import { jobDetail } from "@/test/api-fixtures"
import { recordFetch } from "@/test/fetch-recorder"
import { getFormStore } from "@/ui/form-store"
import { tabOwners } from "@/ui/tab"
import { cancelEdit, enqueueForm, useQueueEdit } from "./edit-session"
import { EnqueueButton, QueueEditBanner } from "./FormQueueControls"

configure({ asyncUtilTimeout: 5_000 })

const schema = tabSchema("txt2img")
const renderWithClient = (ui: React.ReactNode) => render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>)

beforeEach(() => {
  cancelEdit("txt2img")
  useQueueEdit.setState({ saving: {} })
  queryClient.setQueryData(keys.schema("txt2img"), schema)
  getFormStore("txt2img").getState().load(tabOwners(schema), { conflictParams: schema.conflict_params })
})

afterEach(() => {
  vi.restoreAllMocks()
  queryClient.clear()
})

describe("Add to queue", { timeout: 20_000 }, () => {
  it("queues the form, is busy until the server answered, and reports the queue position", async () => {
    let answer!: () => void
    const answered = new Promise<void>((resolve) => (answer = resolve))
    const success = vi.spyOn(toast, "success")
    const fetch = recordFetch(() => ({ status: 201, body: { job_id: "j9", job_ids: ["j9"], seq: 9 } }))
    fetch.spy.mockImplementationOnce(async (input) => {
      await answered
      const request = input as Request
      fetch.requests.push({ method: request.method, path: new URL(request.url).pathname, query: new URLSearchParams(), body: await request.json(), contentType: null })
      return new Response(JSON.stringify({ job_id: "j9", job_ids: ["j9"], seq: 9 }), { status: 201, headers: { "Content-Type": "application/json" } })
    })
    queryClient.setQueryData(keys.queueItems, [{}, {}])
    getFormStore("txt2img").getState().setValue("core", "prompt", "a lighthouse")

    renderWithClient(<EnqueueButton tab="txt2img" />)
    const button = screen.getByRole("button", { name: "Add to queue" })
    fireEvent.click(button)
    expect(button).toBeDisabled()
    answer()
    await waitFor(() => expect(button).toBeEnabled())
    expect(fetch.requests).toHaveLength(1)
    expect(fetch.last()).toMatchObject({ method: "POST", path: "/api/v2/jobs", body: { tab: "txt2img", core: { prompt: "a lighthouse" } } })
    expect(success).toHaveBeenCalledWith("Added to the queue as #9", expect.objectContaining({ description: "2 jobs were waiting" }))
  })

  it("marks the values the server rejects and explains other failures", async () => {
    recordFetch(() => ({ status: 422, body: { errors: [{ owner: "core", param: "prompt", message: "too long" }] } }))
    expect(await enqueueForm("txt2img")).toBeNull()
    expect(getFormStore("txt2img").getState().errors.core?.prompt).toBe("too long")

    vi.restoreAllMocks()
    const failure = vi.spyOn(toast, "error")
    recordFetch(() => ({ status: 500, body: { detail: "Traceback (most recent call last): ..." } }))
    expect(await enqueueForm("txt2img")).toBeNull()
    // a server error is described in the UI's words, not with the server's traceback
    expect(failure).toHaveBeenCalledWith("Could not queue the job", { description: "The server ran into an error (500)." })
  })

  it("does nothing before the form's schema is loaded", async () => {
    const fetch = recordFetch()
    queryClient.removeQueries({ queryKey: keys.schema("txt2img") })
    expect(await enqueueForm("txt2img")).toBeNull()
    expect(fetch.requests).toEqual([])
  })
})

describe("queued-job edit banner", { timeout: 20_000 }, () => {
  it("is hidden unless a queued job is being edited", () => {
    const { container } = renderWithClient(<QueueEditBanner tab="txt2img" />)
    expect(container).toBeEmptyDOMElement()
  })

  it("names the job, updates it with the form, and stops editing", async () => {
    useQueueEdit.setState({ sessions: { txt2img: { jobId: "j1", seq: 12, tab: "txt2img", title: "a lighthouse at dusk" } } })
    const fetch = recordFetch(() => ({ body: jobDetail("j1", "queued") }))
    renderWithClient(<QueueEditBanner tab="txt2img" />)
    expect(screen.getByRole("status")).toHaveTextContent("Editing queued job #12")
    expect(screen.getByText("a lighthouse at dusk")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Update job" }))
    await waitFor(() => expect(screen.queryByRole("status")).toBeNull())
    expect(fetch.last()).toMatchObject({ method: "PUT", path: "/api/v2/jobs/j1/params" })
    expect(useQueueEdit.getState().sessions.txt2img).toBeUndefined()
  })

  it("disables Update while saving, and Stop editing leaves the job as it was", () => {
    const fetch = recordFetch()
    useQueueEdit.setState({ sessions: { txt2img: { jobId: "j1", seq: 3, tab: "txt2img", title: "" } }, saving: { txt2img: true } })
    renderWithClient(<QueueEditBanner tab="txt2img" />)
    expect(screen.getByRole("button", { name: "Update job" })).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Stop editing" }))
    expect(screen.queryByRole("status")).toBeNull()
    expect(useQueueEdit.getState().sessions.txt2img).toBeUndefined()
    expect(fetch.requests).toEqual([])
  })
})
