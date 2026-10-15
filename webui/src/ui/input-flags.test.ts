/**
 * Input flags in the form store: `locks` (programmatic writes skip the
 * locked param), `persist: false` (never drafted), `submit: false` (never
 * sent with a job) and image lists with a widget hint (upload semantics).
 */
import i18next from "i18next"
import { afterEach, describe, expect, it } from "vitest"
import { createFormStore, draftKey, type FormStore } from "./form-store"
import type { OwnerSpec } from "./form-index"
import { messageText } from "./messages"
import type { TabSchema } from "./schema"
import { jobRequest } from "./tab"

const owners: OwnerSpec[] = [
  {
    owner: "core",
    layout: { type: "column", children: [{ type: "text", name: "prompt", label: "Prompt" }] },
    defaults: { prompt: "" },
  },
  {
    owner: "sampler",
    layout: { type: "slider", name: "steps", label: "Steps", minimum: 1, maximum: 150, step: 1, integer: true },
    defaults: { steps: 20 },
  },
  {
    owner: "lock_steps",
    layout: { type: "toggle", name: "lock", label: "Lock", locks: "sampler.steps" },
    defaults: { lock: false },
  },
  {
    owner: "notes",
    layout: {
      type: "group",
      children: [
        { type: "toggle", name: "pin", label: "Pin", locks: "text" },
        { type: "text", name: "text", label: "Text", persist: false, submit: false },
        { type: "slider", name: "size", label: "Size", minimum: 0, maximum: 10, step: 1, persist: false },
        { type: "image_list", name: "refs", label: "Refs", widget: "image-list", props: {} },
      ],
    },
    defaults: { pin: false, text: "from file", size: 5, refs: [] },
  },
]

let forms = 0
function form(id = `flags-${++forms}`, persistDrafts = false): FormStore {
  const store = createFormStore(id, { persistDrafts })
  store.getState().load(owners)
  return store
}

const steps = (store: FormStore) => store.getState().values.sampler?.steps

afterEach(() => localStorage.clear())

describe("locks", () => {
  it("skip pasted values and patch values of the locked param while the lock is on", () => {
    const store = form()
    store.getState().setValue("lock_steps", "lock", true)
    store.getState().setValues("sampler", { steps: 50 })
    store.getState().setValues("core", { prompt: "cat" })
    store.getState().applyPatches([
      ["sampler", "steps", { value: 60, maximum: 200 }],
      ["core", "prompt", { value: "dog" }],
    ])
    expect(steps(store)).toBe(20)
    expect(store.getState().values.core?.prompt).toBe("dog")
    // Display parts of a patch still apply
    expect(store.getState().overrides.sampler?.steps).toEqual({ maximum: 200 })

    store.getState().setValue("lock_steps", "lock", false)
    store.getState().setValues("sampler", { steps: 50 })
    expect(steps(store)).toBe(50)
  })

  it("let user edits and resets through", () => {
    const store = form()
    store.getState().setValue("lock_steps", "lock", true)
    store.getState().setValue("sampler", "steps", 33)
    expect(steps(store)).toBe(33)
    store.getState().reset("sampler")
    expect(steps(store)).toBe(20)
  })

  it("resolve a bare name to the lock's own owner", () => {
    const store = form()
    store.getState().setValues("notes", { pin: true })
    store.getState().setValues("notes", { text: "pasted", size: 9 })
    expect(store.getState().values.notes).toMatchObject({ text: "from file", size: 9 })
  })

  it("hold without any component mounted, for writes before the page opens", () => {
    // e.g. "Send to img2img" delivered into a loaded store that is not on screen
    const store = form()
    store.getState().setValues("lock_steps", { lock: true })
    store.getState().setValues("sampler", { steps: 7 })
    expect(steps(store)).toBe(20)
  })
})

describe("persist: false", () => {
  it("is never written to drafts", async () => {
    const store = form("flags-drafted", true)
    store.getState().setValue("notes", "text", "edited")
    store.getState().setValue("notes", "size", 8)
    store.getState().setValue("core", "prompt", "kept")
    await new Promise((resolve) => setTimeout(resolve, 400))
    const saved = JSON.parse(localStorage.getItem(draftKey("flags-drafted")) ?? "{}")
    expect(saved.state.values.core.prompt).toBe("kept")
    expect(saved.state.values.notes).not.toHaveProperty("text")
    expect(saved.state.values.notes).not.toHaveProperty("size")
    expect(saved.state.values.notes).not.toHaveProperty("refs")
  })

  it("starts from the current default on a page load, even with an older draft", () => {
    localStorage.setItem(
      draftKey("flags-old"),
      JSON.stringify({ state: { values: { notes: { text: "stale", size: 1, refs: ["gone-1"], pin: true } }, open: {}, tabs: {} }, version: 1 }),
    )
    const store = createFormStore("flags-old")
    store.getState().load(owners)
    // Upload ids of a widget-edited image list are not restored either
    expect(store.getState().values.notes).toEqual({ pin: true, text: "from file", size: 5, refs: [] })
  })

  it("keeps the session's value when the schema reloads", () => {
    const store = form()
    store.getState().setValue("notes", "size", 8)
    store.getState().load(owners)
    expect(store.getState().values.notes?.size).toBe(8)
  })
})

describe("submit: false", () => {
  it("is left out of the job body", () => {
    const store = form()
    const schema = {
      version: 1,
      tab: "txt2img",
      core: { owner: "core", layout: owners[0]!.layout!, defaults: owners[0]!.defaults },
      scripts: owners.slice(1).map((spec) => ({ id: spec.owner, title: spec.owner, kind: "alwayson" as const, layout: spec.layout, defaults: spec.defaults })),
    } satisfies TabSchema
    const body = jobRequest(store, schema)
    expect(body.scripts.notes).toEqual({ pin: false, size: 5, refs: [] })
    expect(body.scripts.sampler).toEqual({ steps: 20 })
    expect(body.core).toEqual({ prompt: "" })
  })
})

describe("messageText", () => {
  const i18n = i18next.createInstance()
  void i18n.init({
    lng: "zh-CN",
    resources: { "zh-CN": { translation: { "Saved.": "已保存。", "Removed {{count}} tag(s): {{tags}}": "已移除 {{count}} 个标签：{{tags}}" } } },
    keySeparator: false,
    nsSeparator: false,
    interpolation: { escapeValue: false },
  })

  it("translates catalog strings and messages, and leaves other text verbatim", () => {
    expect(messageText(i18n, "Saved.")).toBe("已保存。")
    expect(messageText(i18n, { text: "Removed {{count}} tag(s): {{tags}}", params: { count: 2, tags: "a, b" } })).toBe("已移除 2 个标签：a, b")
    expect(messageText(i18n, { text: "Unknown {{n}}", params: { n: 1 } })).toBe("Unknown 1")
    expect(messageText(i18n, "Not in a catalog: {{x}}")).toBe("Not in a catalog: {{x}}")
  })
})
