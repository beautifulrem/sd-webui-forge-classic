import { afterEach, describe, expect, it, vi } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import * as actions from "@/ui/actions"
import { buildIndex } from "@/ui/form-index"
import { createFormStore } from "@/ui/form-store"
import { tabOwners } from "@/ui/tab"
import { splitInfotext } from "./display"
import { applyPaste, pastePlan, planEntries } from "./paste"

function loadedStore() {
  const schema = tabSchema("txt2img")
  const store = createFormStore("paste-test", { persistDrafts: false })
  store.getState().load(tabOwners(schema), { conflictParams: schema.conflict_params })
  return store
}

const parsed = {
  core: { prompt: "a cat", cfg_scale: 4, width: 832, height: 1216, unknown_param: 1 },
  scripts: { sampler: { steps: 30, sampler_name: "DPM++ 2M" }, seed: { seed: 42 }, gone: { x: 1 } },
}

afterEach(() => vi.restoreAllMocks())

describe("pastePlan", () => {
  const index = buildIndex(tabOwners(tabSchema("txt2img")))

  it("keeps the params the form has and lists the others", () => {
    const plan = pastePlan(index, parsed)
    expect(plan.entries).toEqual([
      { owner: "core", values: { prompt: "a cat", cfg_scale: 4, width: 832, height: 1216 } },
      { owner: "sampler", values: { steps: 30, sampler_name: "DPM++ 2M" } },
      { owner: "seed", values: { seed: 42 } },
    ])
    expect(plan.skipped).toEqual(["core.unknown_param", "gone.x"])
  })

  it("drops seed and size for send_seed / send_size off", () => {
    const plan = pastePlan(index, parsed, { seed: false, size: false })
    expect(plan.entries).toEqual([
      { owner: "core", values: { prompt: "a cat", cfg_scale: 4 } },
      { owner: "sampler", values: { steps: 30, sampler_name: "DPM++ 2M" } },
    ])
  })

  it("plans raw entries the same way", () => {
    expect(planEntries(index, [{ owner: "core", values: { init_img: "u1", width: 640 } }]).entries).toEqual([{ owner: "core", values: { width: 640 } }])
  })
})

describe("applyPaste", () => {
  it("sets the values, then runs each on_change once and one conflicts pass", () => {
    const run = vi.spyOn(actions, "runAction").mockResolvedValue(null)
    const store = loadedStore()
    const ctx = { store, tab: "txt2img" }
    const count = applyPaste(ctx, [
      { owner: "core", values: { width: 640, height: 640, enable_hr: true } },
      { owner: "sampler", values: { sampler_name: "LCM" } },
    ])
    expect(count).toBe(4)
    expect(store.getState().values.core).toMatchObject({ width: 640, height: 640, enable_hr: true })
    expect(store.getState().values.sampler?.sampler_name).toBe("LCM")
    // width, height and enable_hr share `hr_resolution`: one call. Conflicts: one bulk call, no `selected`.
    expect(run.mock.calls.map(([, owner, action, extra]) => [owner, action, extra])).toEqual([
      ["core", "hr_resolution", undefined],
      ["core", "__conflicts__", undefined],
    ])
  })
})

describe("splitInfotext", () => {
  it("splits prompt, negative prompt and the parameter line", () => {
    const parts = splitInfotext('1girl, (smile:1.2)\nsecond line\nNegative prompt: lowres\nbad hands\nSteps: 20, Sampler: Euler a, Lora hashes: "a: 1, b: 2", Size: 512x768')
    expect(parts.prompt).toBe("1girl, (smile:1.2)\nsecond line")
    expect(parts.negative).toBe("lowres\nbad hands")
    expect(parts.params).toEqual([
      ["Steps", "20"],
      ["Sampler", "Euler a"],
      ["Lora hashes", "a: 1, b: 2"],
      ["Size", "512x768"],
    ])
  })

  it("keeps a prompt without parameters whole", () => {
    expect(splitInfotext("just a prompt")).toEqual({ prompt: "just a prompt", negative: "", params: [] })
  })
})
