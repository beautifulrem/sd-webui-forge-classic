import { beforeEach, describe, expect, it } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import type { Job } from "@/api/types"
import { getFormStore } from "@/ui/form-store"
import { tabOwners } from "@/ui/tab"
import { jobsAhead, resetLosesWork } from "./run-helpers"
import { jobOf } from "@/test/api-fixtures"

const queued = (id: string): Job => jobOf(id, "queued")

describe("queue position", () => {
  it("counts the jobs ahead in run order, plus the running one", () => {
    // Run order from /queue/items (a high-priority job jumped ahead of older ones).
    const items = [queued("urgent"), queued("old"), queued("mine")]
    expect(jobsAhead(items, "mine", true)).toBe(3)
    expect(jobsAhead(items, "urgent", false)).toBe(0)
    expect(jobsAhead(items, "gone", true)).toBeNull()
  })
})

describe("reset confirmation", () => {
  beforeEach(() => {
    for (const tab of ["txt2img", "img2img"] as const) getFormStore(tab).getState().load(tabOwners(tabSchema(tab)))
  })

  it("asks only when prompts or source images would be lost", () => {
    const store = getFormStore("txt2img")
    store.getState().setValues("core", { prompt: "", negative_prompt: "  " })
    expect(resetLosesWork("txt2img")).toBe(false)
    store.getState().setValues("core", { negative_prompt: "lowres" })
    expect(resetLosesWork("txt2img")).toBe(true)

    const img = getFormStore("img2img")
    img.getState().setValues("core", { prompt: "", negative_prompt: "", init_img: null })
    expect(resetLosesWork("img2img")).toBe(false)
    img.getState().setValues("core", { init_img: "upload-1" })
    expect(resetLosesWork("img2img")).toBe(true)
  })
})
