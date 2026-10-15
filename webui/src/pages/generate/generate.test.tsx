import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { useProgress } from "@/api/progress"
import { keys } from "@/api/queries"
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
    const [status, data] = path === "/jobs" ? jobs(body) : [200, {}]
    return new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json" } })
  })
  return calls
}

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  client.setQueryData(keys.schema("txt2img"), tabSchema("txt2img"))
  render(
    <QueryClientProvider client={client}>
      <GeneratePage route={{ name: "generate", tab: "txt2img" }} />
    </QueryClientProvider>,
  )
}

const generateButton = () => screen.getByRole("button", { name: /^Generate/ })

describe("generate page", () => {
  beforeEach(() => useProgress.setState({ running: null, owned: {}, last: {}, orphans: {} }))
  afterEach(() => vi.restoreAllMocks())

  it("submits the form and tracks the job as this form's", async () => {
    const calls = mockServer(() => [200, { job_id: "job-7" }])
    renderPage()
    fireEvent.click(generateButton())
    await waitFor(() => expect(useProgress.getState().owned).toEqual({ "job-7": "txt2img" }))
    const body = calls.find((c) => c.path === "/jobs")?.body as { tab: string; scripts: Record<string, unknown> }
    expect(body.tab).toBe("txt2img")
    expect(body.scripts.sampler).toMatchObject({ sampler_name: "Euler a" })
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
  })

  it("Ctrl/Cmd+Enter generates once per press; Escape never interrupts", async () => {
    const calls = mockServer(() => [200, { job_id: "job-1" }])
    renderPage()
    fireEvent.keyDown(window, { key: "Enter", ctrlKey: true })
    fireEvent.keyDown(window, { key: "Enter", ctrlKey: true, repeat: true })
    fireEvent.keyDown(window, { key: "Enter", ctrlKey: true, repeat: true })
    await waitFor(() => expect(calls.filter((c) => c.path === "/jobs")).toHaveLength(1))

    act(() => useProgress.setState({ running: { jobId: "job-1", step: 1, steps: 20 } }))
    fireEvent.keyDown(window, { key: "Escape" })
    expect(calls.some((c) => c.path.endsWith("/interrupt"))).toBe(false)
    fireEvent.click(screen.getByRole("button", { name: "Interrupt" }))
    await waitFor(() => expect(calls.map((c) => `${c.method} ${c.path}`)).toContain("POST /jobs/job-1/interrupt"))
  })
})
