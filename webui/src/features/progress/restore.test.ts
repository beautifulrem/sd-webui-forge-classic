import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import { useProgress } from "@/api/progress"
import { keys, queryClient } from "@/api/queries"
import type { JobDetail } from "@/api/types"
import { useWs } from "@/api/ws"
import { tabSchema } from "@/api/mock/fixtures"
import { useSession } from "@/features/gallery/session-store"
import { FakeSocket } from "@/test/fake-socket"
import { resetRestore, restoreRuns, wireRunSync } from "./restore"
import { dispatchRun, persistedJobs, persistedRun, persistKey, resetRunControllers, useRuns, wireRunControllers } from "./run-controller"
import { phaseOf } from "./run-machine"
import { created, jobDetail } from "@/test/api-fixtures"

const job = jobDetail

/** `GET /jobs/{id}` answering from `jobs` (404 for unknown ids). */
function serverJobs(jobs: Record<string, JobDetail>) {
  return vi.spyOn(api.jobs, "get").mockImplementation(async (id) => {
    const found = jobs[id]
    if (!found) throw new ApiError(404, "not found")
    return found
  })
}

async function socket() {
  useWs.getState().connect()
  await vi.waitFor(() => expect(FakeSocket.instances.length).toBeGreaterThan(0))
  const ws = FakeSocket.instances.at(-1)!
  ws.open()
  return ws
}

const send = (ws: FakeSocket, event: object) => ws.receive(JSON.stringify(event))

beforeEach(() => {
  FakeSocket.instances = []
  vi.stubGlobal("WebSocket", FakeSocket)
  resetRunControllers()
  resetRestore()
  wireRunControllers()
  wireRunSync()
  useSession.getState().clear()
  useProgress.setState({ running: null, owned: {}, last: {}, orphans: {} })
  queryClient.setQueryData(keys.schema("txt2img"), tabSchema("txt2img"))
})
afterEach(() => {
  useWs.getState().disconnect()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe("restoreRuns", () => {
  it("re-attaches queued/running jobs and files finished ones", async () => {
    // A previous page load left three jobs in flight.
    dispatchRun("txt2img", { type: "submitted", jobIds: ["a", "b", "c"] })
    expect(persistedJobs("txt2img")).toEqual(["a", "b", "c"])
    resetRunControllers()
    localStorage.setItem(persistKey("txt2img"), JSON.stringify(["a", "b", "c"]))

    const get = serverJobs({
      a: job("a", "completed", [{ id: "out-a", kind: "image", infotext: "a\nSteps: 1, Seed: 1, Size: 8x8" }]),
      b: job("b", "running"),
      c: job("c", "queued"),
    })
    await restoreRuns()
    // Each persisted id is looked up on its own (no history page to fall off).
    expect(get.mock.calls.map(([id]) => id).sort()).toEqual(["a", "b", "c"])
    expect(useRuns.getState().runs.txt2img).toMatchObject({ jobs: ["b", "c"], running: "b" })
    expect(useProgress.getState().owned).toMatchObject({ b: "txt2img", c: "txt2img" })
    expect(useSession.getState().items.map((item) => item.fileId)).toEqual(["out-a"])
    expect(persistedJobs("txt2img")).toEqual(["b", "c"])
  })

  it("keeps Generate forever going across a reload while its job is in flight", async () => {
    // A previous page load was generating forever: its job is still running.
    vi.spyOn(api.jobs, "create").mockResolvedValue(created("f1"))
    dispatchRun("txt2img", { type: "forever", on: true })
    await vi.waitFor(() => expect(persistedRun("txt2img")).toEqual({ jobs: ["f1"], forever: true, repeatLeft: 0 }))
    resetRunControllers()
    localStorage.setItem(persistKey("txt2img"), JSON.stringify({ jobs: ["f1"], forever: true, repeatLeft: 0 }))

    const ws = await socket()
    serverJobs({ f1: job("f1", "running") })
    await restoreRuns()
    // The menu's "Generate forever" is checked again, and the run after this one is queued.
    expect(useRuns.getState().runs.txt2img).toMatchObject({ jobs: ["f1"], running: "f1", forever: true })
    expect(persistedRun("txt2img").forever).toBe(true)
    const create = vi.spyOn(api.jobs, "create").mockResolvedValue(created("f2"))
    send(ws, { type: "job.completed", job_id: "f1", seq: 1, status: "completed", outputs: [] })
    await vi.waitFor(() => expect(create).toHaveBeenCalledTimes(1))
    await vi.waitFor(() => expect(useRuns.getState().runs.txt2img.jobs).toEqual(["f2"]))
  })

  it("does not start Generate forever again when its job ended while the page was closed", async () => {
    localStorage.setItem(persistKey("txt2img"), JSON.stringify({ jobs: ["f1"], forever: true, repeatLeft: 0 }))
    serverJobs({ f1: job("f1", "completed") })
    const create = vi.spyOn(api.jobs, "create")
    await restoreRuns()
    expect(create).not.toHaveBeenCalled()
    expect(useRuns.getState().runs.txt2img).toMatchObject({ jobs: [], forever: false })
    expect(localStorage.getItem(persistKey("txt2img"))).toBeNull()
  })

  it("reads the bare job list an older version persisted", () => {
    localStorage.setItem(persistKey("img2img"), JSON.stringify(["x", 3, "y"]))
    expect(persistedRun("img2img")).toEqual({ jobs: ["x", "y"], forever: false, repeatLeft: 0 })
  })

  it("forgets jobs the server no longer knows", async () => {
    localStorage.setItem(persistKey("img2img"), JSON.stringify(["gone"]))
    serverJobs({})
    await restoreRuns()
    expect(localStorage.getItem(persistKey("img2img"))).toBeNull()
    expect(useRuns.getState().runs.img2img.jobs).toEqual([])
  })

  it("applies a job event that arrives while its lookup is pending", async () => {
    const ws = await socket()
    localStorage.setItem(persistKey("txt2img"), JSON.stringify(["r"]))
    let answer!: (value: JobDetail) => void
    vi.spyOn(api.jobs, "get").mockReturnValue(new Promise((resolve) => (answer = resolve)))
    const restored = restoreRuns()
    // The job finishes before the (stale) lookup answer arrives.
    send(ws, { type: "job.completed", job_id: "r", seq: 1, status: "completed", outputs: [] })
    answer(job("r", "running"))
    await restored
    expect(phaseOf(useRuns.getState().runs.txt2img)).toBe("idle")
    expect(localStorage.getItem(persistKey("txt2img"))).toBeNull()
  })

  it("does not drop a job submitted while the restore is pending", async () => {
    localStorage.setItem(persistKey("txt2img"), JSON.stringify(["old"]))
    let answer!: (value: JobDetail) => void
    vi.spyOn(api.jobs, "get").mockReturnValue(new Promise((resolve) => (answer = resolve)))
    const restored = restoreRuns()
    dispatchRun("txt2img", { type: "submitted", jobIds: ["new"] })
    answer(job("old", "completed"))
    await restored
    expect(useRuns.getState().runs.txt2img.jobs).toEqual(["new"])
    expect(persistedJobs("txt2img")).toEqual(["new"])
  })
})

describe("resync after missed events", () => {
  it("goes idle when the job completed while the socket was down", async () => {
    let ws = await socket()
    vi.spyOn(api.jobs, "create").mockResolvedValue(created("j1"))
    dispatchRun("txt2img", { type: "forever", on: true })
    await vi.waitFor(() => expect(useRuns.getState().runs.txt2img.jobs).toEqual(["j1"]))
    send(ws, { type: "job.started", job_id: "j1", seq: 1 })
    useProgress.getState().progress({ job_id: "j1", seq: 1, step: 3, steps: 8 })

    // Laptop sleeps: the socket drops and the job fails meanwhile (so forever stops).
    ws.drop()
    serverJobs({ j1: job("j1", "failed") })
    ws = await socket()
    send(ws, { type: "hello", features: [] })
    send(ws, { type: "queue.changed", size: 0, paused: false, running: null })

    await vi.waitFor(() => expect(phaseOf(useRuns.getState().runs.txt2img)).toBe("idle"))
    expect(useRuns.getState().runs.txt2img.forever).toBe(false)
    expect(useProgress.getState().running).toBeNull()
    expect(localStorage.getItem(persistKey("txt2img"))).toBeNull()
  })

  it("files the results of a job that completed unseen and keeps generating forever", async () => {
    const ws = await socket()
    const create = vi.spyOn(api.jobs, "create").mockResolvedValueOnce(created("f1")).mockResolvedValueOnce(created("f2"))
    dispatchRun("txt2img", { type: "forever", on: true })
    await vi.waitFor(() => expect(useRuns.getState().runs.txt2img.jobs).toEqual(["f1"]))
    send(ws, { type: "job.started", job_id: "f1", seq: 1 })
    serverJobs({ f1: job("f1", "completed", [{ id: "out-f1", kind: "image" }]) })
    // A queue change naming another running job reveals the missed completion.
    send(ws, { type: "queue.changed", size: 0, running: "someone-else" })
    await vi.waitFor(() => expect(create).toHaveBeenCalledTimes(2))
    expect(useSession.getState().items.map((item) => item.fileId)).toEqual(["out-f1"])
    await vi.waitFor(() => expect(useRuns.getState().runs.txt2img.jobs).toEqual(["f2"]))
  })

  it("marks a queued job running when the queue says it runs", async () => {
    const ws = await socket()
    dispatchRun("img2img", { type: "submitted", jobIds: ["q1"] })
    send(ws, { type: "queue.changed", size: 0, running: "q1" })
    expect(phaseOf(useRuns.getState().runs.img2img)).toBe("running")
  })
})
