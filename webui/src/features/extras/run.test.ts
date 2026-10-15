import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { extrasSchema } from "@/api/mock/fixtures"
import { useProgress } from "@/api/progress"
import { keys, queryClient } from "@/api/queries"
import { getFormStore } from "@/ui/form-store"
import { tabOwners } from "@/ui/tab"
import { finishExtras, interruptExtras, resetExtrasRuns, runExtras, singleSource, useExtrasRuns, wireExtrasRuns } from "./run"

function mockFetch(answer: (method: string, path: string) => [number, unknown]) {
  const calls: { method: string; path: string; body: unknown }[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const request = input as Request
    const path = new URL(request.url).pathname.replace("/api/v2", "")
    const text = request.method === "GET" ? "" : await request.text()
    calls.push({ method: request.method, path, body: text ? JSON.parse(text) : undefined })
    const [status, data] = answer(request.method, path)
    return new Response(status === 204 ? null : JSON.stringify(data), { status, headers: { "Content-Type": "application/json" } })
  })
  return calls
}

describe("extras runs", () => {
  beforeEach(() => {
    resetExtrasRuns()
    useProgress.setState({ running: null, owned: {}, last: {}, orphans: {} })
    queryClient.setQueryData(keys.schema("extras"), extrasSchema)
    getFormStore("extras").getState().load(tabOwners(extrasSchema))
  })
  afterEach(() => vi.restoreAllMocks())

  it("knows the single source image", () => {
    expect(singleSource({ mode: "Single Image", image: "up1" })).toBe("up1")
    expect(singleSource({ image: "up1" })).toBe("up1")
    expect(singleSource({ mode: "Batch Process", image: "up1" })).toBeNull()
    expect(singleSource({ mode: "Single Image", image: null })).toBeNull()
  })

  it("submits the form, follows the job and keeps its source for the comparison", async () => {
    wireExtrasRuns()
    const calls = mockFetch(() => [201, { job_id: "x1", job_ids: ["x1"], seq: 3 }])
    getFormStore("extras").getState().setValue("core", "image", "up9")
    await runExtras()
    const body = calls[0]!.body as { tab: string; core: { image: string } }
    expect(body.tab).toBe("extras")
    expect(body.core.image).toBe("up9")
    expect(useExtrasRuns.getState().jobs).toEqual(["x1"])
    expect(useProgress.getState().owned).toEqual({ x1: "extras" })

    // The WebSocket bridge reports the result through the progress store.
    useProgress.getState().completed({ job_id: "x1", outputs: [{ id: "out1", kind: "image" }], infotext: "Postprocess upscale by: 4" })
    const [result] = useExtrasRuns.getState().results
    expect(result).toMatchObject({ jobId: "x1", source: "up9", info: "Postprocess upscale by: 4" })
    expect(result!.outputs.map((output) => output.id)).toEqual(["out1"])
    expect(useExtrasRuns.getState().jobs).toEqual([])
  })

  it("marks rejected values on the form", async () => {
    mockFetch(() => [422, { errors: [{ owner: "core", param: "image", message: "an image is required" }] }])
    await runExtras()
    expect(getFormStore("extras").getState().errors.core?.image).toBe("an image is required")
    expect(useExtrasRuns.getState().jobs).toEqual([])
  })

  it("interrupts the running job and cancels the queued ones", async () => {
    const calls = mockFetch(() => [200, {}])
    useExtrasRuns.setState({ jobs: ["a", "b"] })
    useProgress.setState({ running: { jobId: "a", step: 0, steps: 0 } })
    interruptExtras()
    await vi.waitFor(() => expect(calls.map((call) => `${call.method} ${call.path}`).sort()).toEqual(["DELETE /jobs/b", "POST /jobs/a/interrupt"]))
  })

  it("ignores jobs it did not queue", () => {
    finishExtras({ jobId: "other", outputs: [] })
    expect(useExtrasRuns.getState().results).toEqual([])
  })
})
