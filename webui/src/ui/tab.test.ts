import { describe, expect, it } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { createFormStore } from "./form-store"
import { jobRequest, missingInputs, tabOwners } from "./tab"

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
    // Hidden state and JSON values are submitted as they are.
    expect(body.scripts.anima?.internal_state).toEqual({ version: 2 })
    expect(body.scripts["misc-ext"]).toEqual({ flag: true, pair: [0.5, "auto"] })
  })
})

describe("missingInputs", () => {
  it("lists the required core inputs a body leaves empty, as the server would", () => {
    const schema = tabSchema("img2img")
    expect(missingInputs(schema, { core: { mode: "img2img", init_img: null } })).toEqual([{ owner: "core", param: "init_img", message: "an image is required" }])
    expect(missingInputs(schema, { core: { mode: "Inpaint", init_img: "up-1" } })).toEqual([{ owner: "core", param: "mask", message: "paint a mask over the image" }])
    expect(missingInputs(schema, { core: { mode: "Inpaint", init_img: "up-1", mask: "up-2" } })).toEqual([])
    // a left-out name takes its default (no image); blank strings and empty lists are empty
    expect(missingInputs(schema, { core: { mode: "Sketch" } }).map((e) => e.param)).toEqual(["init_img"])
    const extras = { ...schema, core: { ...schema.core, required: [{ param: "dir", message: "an input directory is required", when: true }] } }
    expect(missingInputs(extras, { core: { dir: "  " } })).toHaveLength(1)
    expect(missingInputs(tabSchema("txt2img"), { core: {} })).toEqual([])
  })
})
