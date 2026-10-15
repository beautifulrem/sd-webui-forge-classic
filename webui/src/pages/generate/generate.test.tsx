import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { useProgress } from "@/api/progress"
import { keys, queryClient } from "@/api/queries"
import { useSession } from "@/features/gallery/session-store"
import { dispatchRun, resetRunControllers, useRuns } from "@/features/progress/run-controller"
import { getFormStore } from "@/ui/form-store"
import GeneratePage from "./index"

type Call = { path: string; method: string; body: unknown }

function mockServer(jobs: (body: unknown) => [number, unknown]) {
  const calls: Call[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const request = input as Request
    const path = new URL(request.url).pathname.replace("/api/v2", "")
    const body = request.method === "POST" && path === "/jobs" ? await request.json() : undefined
    calls.push({ path, method: request.method, body })
    const [status, data] = path === "/jobs" && request.method === "POST" ? jobs(body) : [200, path === "/jobs" ? { items: [] } : {}]
    return new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json" } })
  })
  return calls
}

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const schema = tabSchema("txt2img")
  client.setQueryData(keys.schema("txt2img"), schema)
  client.setQueryData(keys.schema("img2img"), tabSchema("img2img"))
  // The run controller reads the schema from the app's query client.
  queryClient.setQueryData(keys.schema("txt2img"), schema)
  render(
    <QueryClientProvider client={client}>
      <GeneratePage route={{ name: "generate", tab: "txt2img" }} />
    </QueryClientProvider>,
  )
}

/** The main Generate button ("Generate" + its shortcut hint), not "Generate several times". */
const generateButton = () => screen.getAllByRole("button", { name: /^Generate/ }).find((b) => !b.hasAttribute("aria-label"))!

const posted = (calls: Call[], path: string) => calls.filter((c) => c.method === "POST" && c.path === path)

describe("generate page", () => {
  beforeEach(() => {
    useProgress.setState({ running: null, owned: {}, last: {}, orphans: {} })
    useSession.getState().clear()
    resetRunControllers()
  })
  afterEach(() => vi.restoreAllMocks())

  it("submits the form and tracks the job as this form's", async () => {
    const calls = mockServer(() => [200, { job_id: "job-7" }])
    renderPage()
    fireEvent.click(generateButton())
    await waitFor(() => expect(useProgress.getState().owned).toEqual({ "job-7": "txt2img" }))
    const body = posted(calls, "/jobs")[0]?.body as { tab: string; scripts: Record<string, unknown> }
    expect(body.tab).toBe("txt2img")
    expect(body.scripts.sampler).toMatchObject({ sampler_name: "Euler a" })
    expect(useRuns.getState().runs.txt2img.jobs).toEqual(["job-7"])
  })

  it("marks rejected params on their fields and opens what hides them", async () => {
    mockServer(() => [
      422,
      {
        errors: [
          { owner: "core", param: "hr_scale", message: "must be at most 4" },
          { owner: "sampler", param: "steps", message: "too many steps" },
        ],
      },
    ])
    renderPage()
    fireEvent.click(generateButton())
    expect(await screen.findByText("must be at most 4")).toBeInTheDocument() // inside the (closed) Hires. fix section
    expect(screen.getByText("too many steps")).toBeInTheDocument()
    expect(screen.getByLabelText("Sampling steps", { selector: "input:not([type=range])" })).toHaveAttribute("aria-invalid", "true")
    expect(getFormStore("txt2img").getState().errors.sampler).toEqual({ steps: "too many steps" })
    expect(useRuns.getState().runs.txt2img.pending).toBe(0)
  })

  it("Ctrl/Cmd+Enter generates once per press; Escape interrupts unless it belongs to a field", async () => {
    const calls = mockServer(() => [200, { job_id: "job-1" }])
    renderPage()
    fireEvent.keyDown(window, { key: "Enter", ctrlKey: true })
    fireEvent.keyDown(window, { key: "Enter", ctrlKey: true, repeat: true })
    fireEvent.keyDown(window, { key: "Enter", ctrlKey: true, repeat: true })
    await waitFor(() => expect(useRuns.getState().runs.txt2img.jobs).toEqual(["job-1"]))
    expect(posted(calls, "/jobs")).toHaveLength(1)

    act(() => dispatchRun("txt2img", { type: "started", jobId: "job-1" }))
    // Escape in a text field is the field's.
    const input = document.createElement("input")
    document.body.append(input)
    fireEvent.keyDown(input, { key: "Escape" })
    expect(calls.some((c) => c.path.endsWith("/interrupt"))).toBe(false)
    input.remove()

    fireEvent.keyDown(document.body, { key: "Escape" })
    await waitFor(() => expect(posted(calls, "/jobs/job-1/interrupt")).toHaveLength(1))
    expect(useRuns.getState().runs.txt2img.interrupting).toBe(true)
  })

  it("Alt+Enter skips the running job; the Interrupt button interrupts", async () => {
    const calls = mockServer(() => [200, { job_id: "job-2" }])
    renderPage()
    fireEvent.click(generateButton())
    await waitFor(() => expect(useRuns.getState().runs.txt2img.jobs).toEqual(["job-2"]))
    act(() => dispatchRun("txt2img", { type: "started", jobId: "job-2" }))
    fireEvent.keyDown(window, { key: "Enter", altKey: true })
    await waitFor(() => expect(posted(calls, "/jobs/job-2/skip")).toHaveLength(1))
    fireEvent.click(screen.getByRole("button", { name: "Interrupt" }))
    await waitFor(() => expect(posted(calls, "/jobs/job-2/interrupt")).toHaveLength(1))
  })

  it("files finished jobs into the history and shows the newest on the stage", async () => {
    mockServer(() => [200, { job_id: "job-3" }])
    renderPage()
    act(() => {
      useProgress.getState().track("job-3", "txt2img")
      useProgress.getState().completed({ job_id: "job-3", outputs: [{ id: "out-a", kind: "image" }, { id: "out-b", kind: "image" }], infotext: "a cat\nSteps: 20, Seed: 42, Size: 512x512" })
    })
    expect(useSession.getState().items.map((item) => item.fileId)).toEqual(["out-a", "out-b"])
    const image = await screen.findByRole("img", { name: "Generated image" })
    expect(image.getAttribute("src")).toContain("/files/out-a")
    expect(screen.getByText("a cat")).toBeInTheDocument()
  })
})
