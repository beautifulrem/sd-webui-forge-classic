import { describe, expect, it } from "vitest"
import { normalizeEvent } from "./events"

describe("normalizeEvent", () => {
  it("accepts flat and enveloped events", () => {
    expect(normalizeEvent({ type: "job.progress", job_id: "j", step: 1, steps: 4 })).toEqual({ type: "job.progress", job_id: "j", step: 1, steps: 4 })
    expect(normalizeEvent({ type: "toast", data: { message: "hi", level: "info" } })).toEqual({ type: "toast", message: "hi", level: "info" })
  })

  it("rejects unknown types and events missing required fields", () => {
    expect(normalizeEvent({ type: "nope" })).toBeNull()
    expect(normalizeEvent({ type: "job.progress", job_id: "j", step: "1", steps: 4 })).toBeNull()
    expect(normalizeEvent({ type: "job.started" })).toBeNull()
    expect(normalizeEvent({ type: "hello", features: "all" })).toBeNull()
    expect(normalizeEvent({ type: "toast" })).toBeNull()
    expect(normalizeEvent("text")).toBeNull()
  })

  it("fills optional lists of terminal job events", () => {
    expect(normalizeEvent({ type: "job.completed", job_id: "j" })).toEqual({ type: "job.completed", job_id: "j", outputs: [] })
    expect(normalizeEvent({ type: "job.failed", job_id: "j" })).toMatchObject({ error: "Failed" })
  })
})
