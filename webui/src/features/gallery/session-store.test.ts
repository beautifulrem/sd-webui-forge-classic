import { beforeEach, describe, expect, it } from "vitest"
import { useProgress } from "@/api/progress"
import { noteUpscale, useSession, wireSessionResults } from "./session-store"

const image = (id: string, extra: object = {}) => ({ id, kind: "image" as const, ...extra })
const keys = () => useSession.getState().items.map((item) => item.key)

describe("session results", () => {
  beforeEach(() => {
    useSession.getState().clear()
    useSession.getState().addJob({ jobId: "a", tab: "txt2img", outputs: [image("a0"), image("a1")] })
    useSession.getState().addJob({ jobId: "b", tab: "txt2img", outputs: [image("b0")] })
  })

  it("keeps the role of grids and extra images", () => {
    useSession.getState().addJob({ jobId: "c", tab: "txt2img", outputs: [image("grid", { role: "grid" }), image("c1"), image("map", { role: "extra" })] })
    const [grid, first, map] = useSession.getState().items
    expect([grid!.role, first!.role, map!.role]).toEqual(["grid", undefined, "extra"])
  })

  it("attaches a job's video to its first image", () => {
    useSession.getState().addJob({ jobId: "v", tab: "txt2img", outputs: [image("frame0"), image("frame1"), { id: "clip", kind: "video" }] })
    const [first, second] = useSession.getState().items
    expect(first!.videoId).toBe("clip")
    expect(second!.videoId).toBeUndefined()
    expect(keys()).toEqual(["v:0", "v:1", "b:0", "a:0", "a:1"])
  })

  it("replaces the upscaled result by default (hires_button_gallery_insert off)", () => {
    useSession.getState().addJob({ jobId: "u", tab: "txt2img", outputs: [image("u0")] }, { source: "a:0", insert: false, selectNew: true })
    expect(keys()).toEqual(["b:0", "u:0", "a:1"])
    expect(useSession.getState().selected).toBe("u:0")
  })

  it("inserts next to it and selects the upscaled or the source result (hires_insert_index)", () => {
    useSession.getState().addJob({ jobId: "u", tab: "txt2img", outputs: [image("u0")] }, { source: "a:0", insert: true, selectNew: true })
    expect(keys()).toEqual(["b:0", "a:0", "u:0", "a:1"])
    expect(useSession.getState().selected).toBe("u:0")

    useSession.getState().addJob({ jobId: "w", tab: "txt2img", outputs: [image("w0")] }, { source: "b:0", insert: true, selectNew: false })
    expect(keys()).toEqual(["b:0", "w:0", "a:0", "u:0", "a:1"])
    expect(useSession.getState().selected).toBe("b:0")
  })

  it("puts the result first when the source is gone", () => {
    useSession.getState().addJob({ jobId: "u", tab: "txt2img", outputs: [image("u0")] }, { source: "gone:0", insert: false, selectNew: true })
    expect(keys()[0]).toBe("u:0")
  })

  it("places a finished upscale job noted by the run controller", () => {
    wireSessionResults()
    noteUpscale("job-u", { source: "a:1", insert: false, selectNew: true })
    useProgress.getState().track("job-u", "txt2img")
    useProgress.getState().completed({ job_id: "job-u", seq: 9, tab: "txt2img", status: "completed", outputs: [image("u0")], infotext: "a cat" })
    expect(keys()).toEqual(["b:0", "a:0", "job-u:0"])
  })
})
