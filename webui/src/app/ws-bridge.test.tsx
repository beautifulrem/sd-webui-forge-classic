import { renderHook } from "@testing-library/react"
import { toast } from "sonner"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useProgress } from "@/api/progress"
import { queryClient } from "@/api/queries"
import { useWs } from "@/api/ws"
import { FakeSocket } from "@/test/fake-socket"
import { MODEL_LOADING_TIMEOUT_MS, useWsBridge } from "./ws-bridge"

async function connected() {
  const hook = renderHook(() => useWsBridge())
  await vi.waitFor(() => expect(FakeSocket.instances.length).toBeGreaterThan(0))
  const socket = FakeSocket.instances.at(-1)!
  socket.open()
  const send = (event: object) => socket.receive(JSON.stringify(event))
  send({ type: "hello", features: [] })
  return { hook, socket, send }
}

describe("ws-bridge", () => {
  beforeEach(() => {
    FakeSocket.instances = []
    vi.stubGlobal("WebSocket", FakeSocket)
    useProgress.setState({ running: null, owned: {}, last: {}, orphans: {}, modelLoading: null })
  })
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it("refreshes schemas and pages when the options change", async () => {
    const invalidate = vi.spyOn(queryClient, "invalidateQueries").mockResolvedValue()
    const { hook, send } = await connected()
    send({ type: "options.changed", keys: ["ui_reorder_list"] })
    const keys = invalidate.mock.calls.map(([filters]) => filters?.queryKey)
    expect(keys).toEqual(expect.arrayContaining([["schema"], ["page"], ["options"], ["app"], ["models"], ["styles"]]))
    hook.unmount()
  })

  it("refetches networks on a rescan and everything after a resync", async () => {
    const invalidate = vi.spyOn(queryClient, "invalidateQueries").mockResolvedValue()
    const { hook, send } = await connected()
    send({ type: "networks.changed", kind: "lora" })
    expect(invalidate.mock.calls.map(([filters]) => filters?.queryKey)).toEqual(expect.arrayContaining([["networks"], ["models"]]))
    invalidate.mockClear()
    send({ type: "pong" })
    expect(invalidate).not.toHaveBeenCalled()
    send({ type: "resync", dropped: 300 })
    expect(invalidate).toHaveBeenCalledWith()
    hook.unmount()
  })

  it("clears the previous job's preview when a job starts and keeps results per form", async () => {
    vi.spyOn(queryClient, "invalidateQueries").mockResolvedValue()
    const { hook, send } = await connected()
    useWs.setState({ preview: { jobSeq: 1, step: 20, url: "blob:old" } })
    useProgress.getState().track("j2", "img2img")
    send({ type: "job.started", job_id: "j2", seq: 2, steps: 20 })
    expect(useWs.getState().preview).toBeNull()
    send({ type: "job.completed", job_id: "j2", seq: 2, outputs: [{ id: "f1" }] })
    expect(useProgress.getState().last).toEqual({ img2img: { jobId: "j2", outputs: [{ id: "f1" }], infotext: undefined } })
    // Another client's job never becomes "last".
    send({ type: "job.completed", job_id: "other", outputs: [{ id: "f2" }] })
    expect(useProgress.getState().last.txt2img).toBeUndefined()
    // A job that finishes before its POST answer arrives is attached when tracked.
    send({ type: "job.completed", job_id: "fast", outputs: [{ id: "f3" }] })
    useProgress.getState().track("fast", "txt2img")
    expect(useProgress.getState().last.txt2img?.jobId).toBe("fast")
    hook.unmount()
  })

  it("ends model loading on a terminal stage, model.loaded, a reconnect, or a timeout", async () => {
    vi.spyOn(queryClient, "invalidateQueries").mockResolvedValue()
    vi.useFakeTimers({ shouldAdvanceTime: true })
    const { hook, send } = await connected()
    const loading = () => useProgress.getState().modelLoading
    send({ type: "model.loading", stage: "Loading weights" })
    expect(loading()?.stage).toBe("Loading weights")
    send({ type: "model.loading", stage: "done" })
    expect(loading()).toBeNull()
    send({ type: "model.loading", stage: "Loading weights" })
    send({ type: "model.loaded", model: "x" })
    expect(loading()).toBeNull()
    send({ type: "model.loading", stage: "Loading weights" })
    send({ type: "hello", features: [] })
    expect(loading()).toBeNull()
    send({ type: "model.loading", stage: "Loading weights" })
    vi.advanceTimersByTime(MODEL_LOADING_TIMEOUT_MS + 1)
    expect(loading()).toBeNull()
    hook.unmount()
  })

  it("drops the progress of a job the server no longer runs (missed events)", async () => {
    vi.spyOn(queryClient, "invalidateQueries").mockResolvedValue()
    const { hook, send } = await connected()
    send({ type: "job.progress", job_id: "j1", seq: 1, step: 3, steps: 8 })
    send({ type: "queue.changed", size: 0, running: "j1" })
    expect(useProgress.getState().running?.jobId).toBe("j1")
    // After a reconnect the snapshot says nothing runs any more.
    send({ type: "hello", features: [] })
    send({ type: "queue.changed", size: 0, paused: false, running: null })
    expect(useProgress.getState().running).toBeNull()
    hook.unmount()
  })

  it("does not report a cancelled or interrupted job as failed", async () => {
    vi.spyOn(queryClient, "invalidateQueries").mockResolvedValue()
    const error = vi.spyOn(toast, "error").mockReturnValue(0)
    const { hook, send } = await connected()
    useProgress.getState().track("c1", "txt2img")
    useProgress.getState().track("f1", "txt2img")
    send({ type: "job.failed", job_id: "c1", status: "cancelled", error: "Cancelled" })
    expect(error).not.toHaveBeenCalled()
    send({ type: "job.failed", job_id: "f1", status: "failed", error: "CUDA out of memory" })
    expect(error).toHaveBeenCalledTimes(1)
    hook.unmount()
  })
})
