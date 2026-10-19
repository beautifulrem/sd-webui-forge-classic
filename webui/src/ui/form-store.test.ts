import { describe, expect, it } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { anyModified, createFormStore, draftKey, isModified } from "./form-store"
import type { OwnerSpec } from "./form-index"
import { inputsIn } from "./form-index"
import { tabOwners } from "./tab"

const owners: OwnerSpec[] = [
  {
    owner: "core",
    layout: {
      type: "column",
      children: [
        { type: "slider", name: "steps", label: "Steps", minimum: 1, maximum: 150, step: 1, integer: true },
        { type: "choice", name: "sampler", label: "Sampler", choices: [{ value: "Euler", label: "Euler" }, { value: "DDIM", label: "DDIM" }] },
        { type: "image", name: "init", label: "Init" },
        { type: "text", name: "prompt", label: "Prompt" },
      ],
    },
    defaults: { steps: 20, sampler: "Euler", init: null, prompt: "" },
  },
  {
    owner: "ext",
    layout: {
      type: "section",
      title: "Ext",
      enable: { type: "toggle", name: "enable", label: "Enable" },
      children: [{ type: "multichoice", name: "tags", label: "Tags", choices: [{ value: "a", label: "a" }] }],
    },
    defaults: { enable: false, tags: [] },
  },
]

describe("form store", () => {
  it("loads defaults per owner, including section enable toggles", () => {
    const store = createFormStore("t1", { persistDrafts: false })
    store.getState().load(owners)
    const s = store.getState()
    expect(s.values.core).toEqual({ steps: 20, sampler: "Euler", init: null, prompt: "" })
    expect(s.values.ext).toEqual({ enable: false, tags: [] })
    expect(s.index.inputs.get("ext")?.has("enable")).toBe(true)
  })

  it("tracks modified state and resets fields and sections", () => {
    const store = createFormStore("t2", { persistDrafts: false })
    store.getState().load(owners)
    store.getState().setValue("core", "steps", 30)
    store.getState().setValue("ext", "enable", true)
    store.getState().setValue("ext", "tags", ["a"])
    expect(isModified(store.getState(), "core", "steps")).toBe(true)
    expect(isModified(store.getState(), "core", "sampler")).toBe(false)

    store.getState().reset("core", ["steps"])
    expect(store.getState().values.core!.steps).toBe(20)

    const sectionNames = inputsIn(owners[1]!.layout).map((n) => n.name)
    expect(sectionNames).toEqual(["enable", "tags"])
    expect(anyModified(store.getState(), "ext", sectionNames)).toBe(true)
    store.getState().reset("ext", sectionNames)
    expect(store.getState().values.ext).toEqual({ enable: false, tags: [] })
  })

  it("restores compatible drafts and drops incompatible or upload values", () => {
    localStorage.setItem(
      draftKey("t3"),
      JSON.stringify({
        state: {
          values: {
            core: { steps: 999, sampler: "Gone", init: "upload-1", prompt: "a cat" },
            ext: { enable: "yes", tags: ["a", "zzz"] },
            removed: { x: 1 },
          },
          open: { "ext::enable": true },
          tabs: {},
        },
        version: 1,
      }),
    )
    const store = createFormStore("t3")
    store.getState().load(owners)
    const s = store.getState()
    expect(s.values.core!.steps).toBe(150) // clamped to the slider range
    expect(s.values.core!.sampler).toBe("Euler") // not a choice any more
    expect(s.values.core!.init).toBeNull() // uploads are never restored
    expect(s.values.core!.prompt).toBe("a cat")
    expect(s.values.ext!.enable).toBe(false) // wrong type
    expect(s.values.ext!.tags).toEqual(["a"]) // unknown choice filtered
    expect(s.values.removed).toBeUndefined()
    expect(s.open["ext::enable"]).toBe(true)
  })

  it("persists drafts (debounced) without upload ids", async () => {
    const store = createFormStore("t4")
    store.getState().load(owners)
    store.getState().setValue("core", "prompt", "hello")
    store.getState().setValue("core", "init", "upload-9")
    expect(localStorage.getItem(draftKey("t4"))).toBeNull()
    await new Promise((resolve) => setTimeout(resolve, 400))
    const saved = JSON.parse(localStorage.getItem(draftKey("t4")) ?? "{}")
    expect(saved.state.values.core!.prompt).toBe("hello")
    expect(saved.state.values.core).not.toHaveProperty("init")
  })

  it("applies patches: values and display overrides", () => {
    const store = createFormStore("t5", { persistDrafts: false })
    store.getState().load(owners)
    store.getState().applyPatches([
      ["core", "steps", { value: 40, maximum: 80 }],
      ["ext", "tags", { visible: false, choices: [{ value: "b", label: "B" }] }],
    ])
    const s = store.getState()
    expect(s.values.core!.steps).toBe(40)
    expect(s.overrides.core!.steps).toEqual({ maximum: 80 })
    expect(s.overrides.ext!.tags).toEqual({ visible: false, choices: [{ value: "b", label: "B" }] })
  })

  it("keeps current values when the schema reloads", () => {
    const store = createFormStore("t6", { persistDrafts: false })
    store.getState().load(owners)
    store.getState().setValue("core", "prompt", "keep me")
    store.getState().load(owners)
    expect(store.getState().values.core!.prompt).toBe("keep me")
    expect(store.getState().revision).toBe(2)
  })

  it("loads the full mock tab schema", () => {
    const store = createFormStore("t7", { persistDrafts: false })
    const schema = tabSchema("txt2img")
    store.getState().load(tabOwners(schema))
    const s = store.getState()
    expect(s.values.anima!.preset).toBe("balanced")
    expect(s.values["misc-ext"]?.pair).toEqual([0.5, "auto"])
    expect(s.values.sampler).toEqual({ steps: 28, sampler_name: "Euler a", scheduler: "Automatic" })
    expect(s.index.slots).toEqual(new Set(["sampler", "seed", "accordions", "scripts"]))
  })

  it("rounds and clamps numeric drafts like the server and never restores hidden state", () => {
    const numbers: OwnerSpec[] = [
      {
        owner: "core",
        layout: {
          type: "column",
          children: [
            { type: "slider", name: "steps", label: "Steps", minimum: 1, maximum: 150, step: 1, integer: true },
            { type: "number", name: "shift", label: "Shift", minimum: 0, maximum: 10, precision: 2 },
            { type: "state", name: "internal", label: "" },
          ],
        },
        defaults: { steps: 20, shift: 3, internal: { v: 1 } },
      },
    ]
    localStorage.setItem(
      draftKey("t8"),
      JSON.stringify({ state: { values: { core: { steps: 27.6, shift: 2.34567, internal: { v: 99 } } }, open: {}, tabs: {} }, version: 1 }),
    )
    const store = createFormStore("t8")
    store.getState().load(numbers)
    expect(store.getState().values.core).toEqual({ steps: 28, shift: 2.35, internal: { v: 1 } })
  })

  it("keeps validation errors per field, reveals them and clears them on edit", () => {
    const nested: OwnerSpec[] = [
      {
        owner: "core",
        layout: {
          type: "column",
          children: [
            {
              type: "section",
              title: "Hires",
              children: [
                {
                  type: "tabs",
                  children: [
                    { type: "tab", title: "A", children: [] },
                    { type: "tab", title: "B", children: [{ type: "group", title: "More", open: false, children: [{ type: "text", name: "deep", label: "Deep" }] }] },
                  ],
                },
              ],
            },
            { type: "text", name: "top", label: "Top" },
          ],
        },
        defaults: { deep: "", top: "" },
      },
    ]
    const store = createFormStore("t9", { persistDrafts: false })
    store.getState().load(nested)
    store.getState().setErrors([
      { owner: "core", param: "deep", message: "too deep" },
      { owner: "core", param: "deep", message: "really" },
      { owner: "core", param: "top", message: "bad" },
      { owner: "sampler", param: "steps", message: "no such field here" },
    ])
    const s = store.getState()
    expect(s.errors.core).toEqual({ deep: "too deep; really", top: "bad" })
    expect(s.errors.sampler).toEqual({ steps: "no such field here" })
    // The section, the titled group and the second tab are opened.
    expect(s.open).toEqual({ "core::Hires": true, "core::More::group": true })
    expect(s.tabs).toEqual({ "core::tabs::A|B": "1" })

    store.getState().setValue("core", "top", "fixed")
    expect(store.getState().errors.core).toEqual({ deep: "too deep; really" })
    store.getState().reset("core", ["deep"])
    expect(store.getState().errors.core).toEqual({})
    store.getState().clearErrors()
    expect(store.getState().errors).toEqual({})
  })

  it("does not persist validation errors and clears them on load", () => {
    const store = createFormStore("t10", { persistDrafts: false })
    store.getState().load(owners, { conflictParams: ["core.sampler"] })
    expect(store.getState().conflictParams.has("core.sampler")).toBe(true)
    store.getState().setErrors([{ owner: "core", param: "steps", message: "x" }])
    store.getState().load(owners)
    expect(store.getState().errors).toEqual({})
    expect(store.getState().conflictParams.size).toBe(0)
  })
})
