import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { api } from "@/api/endpoints"
import { useProgress } from "@/api/progress"
import { keys, queryClient } from "@/api/queries"
import { useWs } from "@/api/ws"
import { FakeSocket } from "@/test/fake-socket"
import { useSession, wireSessionResults } from "@/features/gallery/session-store"
import { dispatchRun, persistKey, resetRunControllers, upscaleResult, useRuns, wireRunControllers } from "./run-controller"
import { phaseOf } from "./run-machine"
import { created, jobOf } from "@/test/api-fixtures"
import type { JobCreated } from "@/api/types"

async function socket() {
  useWs.getState().connect()
  await vi.waitFor(() => expect(FakeSocket.instances.length).toBeGreaterThan(0))
  const ws = FakeSocket.instances.at(-1)!
  ws.open()
  return (event: object) => ws.receive(JSON.stringify(event))
}

describe("run controller", () => {
  beforeEach(() => {
    FakeSocket.instances = []
    vi.stubGlobal("WebSocket", FakeSocket)
    resetRunControllers()
    wireRunControllers()
    queryClient.setQueryData(keys.schema("txt2img"), tabSchema("txt2img"))
  })
  afterEach(() => {
    useWs.getState().disconnect()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it("submits, follows the job's events and persists the jobs in flight", async () => {
    const send = await socket()
    vi.spyOn(api.jobs, "create").mockResolvedValue({ job_id: "j1", job_ids: ["j1"], seq: 1 })
    dispatchRun("txt2img", { type: "generate" })
    await vi.waitFor(() => expect(useRuns.getState().runs.txt2img.jobs).toEqual(["j1"]))
    expect(localStorage.getItem(persistKey("txt2img"))).toBe('["j1"]')
    send({ type: "job.started", job_id: "j1", seq: 1 })
    expect(phaseOf(useRuns.getState().runs.txt2img)).toBe("running")
    send({ type: "job.completed", job_id: "j1", seq: 1, status: "completed", outputs: [] })
    expect(phaseOf(useRuns.getState().runs.txt2img)).toBe("idle")
    expect(localStorage.getItem(persistKey("txt2img"))).toBeNull()
  })

  it("does not send a job whose required inputs are empty: marks them and says why", async () => {
    queryClient.setQueryData(keys.schema("img2img"), tabSchema("img2img"))
    const { getFormStore } = await import("@/ui/form-store")
    const { tabOwners } = await import("@/ui/tab")
    const store = getFormStore("img2img")
    store.getState().load(tabOwners(tabSchema("img2img")))
    const create = vi.spyOn(api.jobs, "create")
    const { toast } = await import("sonner")
    const error = vi.spyOn(toast, "error")
    dispatchRun("img2img", { type: "generate" })
    await vi.waitFor(() => expect(store.getState().errors.core?.init_img).toBe("an image is required"))
    expect(create).not.toHaveBeenCalled()
    expect(error).toHaveBeenCalledWith("Some parameters are invalid", { description: "an image is required" })
    expect(phaseOf(useRuns.getState().runs.img2img)).toBe("idle")
  })

  it("upscales a result with the form's values and puts the result in its place", async () => {
    const send = await socket()
    wireSessionResults()
    useSession.getState().clear()
    useSession.getState().addJob({ jobId: "old", tab: "txt2img", outputs: [{ id: "out-1", kind: "image" }, { id: "out-2", kind: "image" }] })
    const create = vi.spyOn(api.jobs, "create").mockResolvedValue({ job_id: "up", job_ids: ["up"], seq: 3 })
    upscaleResult({ fileId: "out-1", key: "old:0" })
    await vi.waitFor(() => expect(useRuns.getState().runs.txt2img.jobs).toEqual(["up"]))
    expect(create.mock.calls[0]![0]).toMatchObject({ tab: "txt2img", upscale_from: "out-1" })
    const completed = { type: "job.completed", job_id: "up", seq: 3, status: "completed", outputs: [{ id: "out-3", kind: "image" as const }] }
    send(completed)
    useProgress.getState().completed(completed) // what app/ws-bridge does
    // hires_button_gallery_insert is off by default: the upscaled image replaces its source
    expect(useSession.getState().items.map((item) => item.fileId)).toEqual(["out-3", "out-2"])
    expect(useSession.getState().selected).toBe("up:0")
  })

  it("replays events of a job that finished before the POST answer", async () => {
    const send = await socket()
    let answer!: (value: JobCreated) => void
    vi.spyOn(api.jobs, "create").mockReturnValue(new Promise((resolve) => (answer = resolve)))
    dispatchRun("txt2img", { type: "generate" })
    send({ type: "job.started", job_id: "fast", seq: 2 })
    send({ type: "job.completed", job_id: "fast", seq: 2, outputs: [] })
    answer(created("fast"))
    await vi.waitFor(() => expect(useRuns.getState().runs.txt2img.pending).toBe(0))
    expect(useRuns.getState().runs.txt2img.jobs).toEqual([])
    expect(phaseOf(useRuns.getState().runs.txt2img)).toBe("idle")
  })

  it("an interrupted job (completed, status interrupted) ends generate forever", async () => {
    const send = await socket()
    vi.spyOn(api.jobs, "create").mockResolvedValue(created("f1"))
    const interrupt = vi.spyOn(api.jobs, "interrupt").mockResolvedValue(jobOf("f1", "interrupted"))
    dispatchRun("txt2img", { type: "forever", on: true })
    await vi.waitFor(() => expect(useRuns.getState().runs.txt2img.jobs).toEqual(["f1"]))
    send({ type: "job.started", job_id: "f1", seq: 3 })
    dispatchRun("txt2img", { type: "interrupt" })
    expect(interrupt).toHaveBeenCalledWith("f1")
    send({ type: "job.completed", job_id: "f1", seq: 3, status: "interrupted", outputs: [] })
    expect(useRuns.getState().runs.txt2img).toMatchObject({ forever: false, jobs: [], pending: 0 })
  })
})
