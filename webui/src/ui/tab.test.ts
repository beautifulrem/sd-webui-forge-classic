import { describe, expect, it } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { createFormStore } from "./form-store"
import { jobRequest, tabOwners } from "./tab"

describe("jobRequest", () => {
  it("submits core values and every script's values, defaults included", () => {
    const schema = tabSchema("txt2img")
    const store = createFormStore("job", { persistDrafts: false })
    store.getState().load(tabOwners(schema))
    store.getState().setValue("sampler", "steps", 12)
    const body = jobRequest(store, schema)
    expect(body.tab).toBe("txt2img")
    expect(body.core).toMatchObject({ width: 1024, enable_hr: false, script: "None" })
    expect(Object.keys(body.scripts)).toEqual(schema.scripts.map((s) => s.id))
    expect(body.scripts.sampler).toEqual({ steps: 12, sampler_name: "Euler a", scheduler: "Automatic" })
    // Hidden state and legacy positional values are submitted as they are.
    expect(body.scripts.anima?.internal_state).toEqual({ version: 2 })
    expect(body.scripts["legacy-ext"]).toEqual({ "#0": true, "#1": [0.5, "auto"] })
  })
})
