import { describe, expect, it } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { createFormStore } from "@/ui/form-store"
import type { TabSchema } from "@/ui/schema"
import { loadSchema } from "./use-form-loader"

/** The img2img mock schema with other core defaults (as after a UI preset switch). */
function withDefaults(schema: TabSchema, defaults: Record<string, unknown>): TabSchema {
  return { ...schema, core: { ...schema.core, defaults: { ...schema.core.defaults, ...defaults } as TabSchema["core"]["defaults"] } }
}

describe("loadSchema", () => {
  it("keeps uploads and lets untouched params follow new defaults on a reload", () => {
    const store = createFormStore("loader-test", { persistDrafts: false })
    const schema = tabSchema("img2img")
    loadSchema(store, schema)
    store.getState().setValues("core", { init_img: "upload-1", cfg_scale: 9 })

    loadSchema(store, withDefaults(tabSchema("img2img"), { width: 512, height: 768, cfg_scale: 7 }))
    const core = store.getState().values.core
    expect(core?.init_img).toBe("upload-1") // `load` alone drops upload ids
    expect(core?.width).toBe(512) // untouched: follows the new default
    expect(core?.height).toBe(768)
    expect(core?.cfg_scale).toBe(9) // edited by the user: kept
  })

  it("does not carry over values on the first load", () => {
    const store = createFormStore("loader-first", { persistDrafts: false })
    loadSchema(store, tabSchema("img2img"))
    expect(store.getState().values.core?.init_img).toBeNull()
  })
})
