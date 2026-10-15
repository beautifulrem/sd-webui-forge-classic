import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import { keys, queryClient } from "@/api/queries"
import type { SettingsSchema } from "@/ui/schema"
import { applyChanges, discardChanges, editorValues, pendingChanges, SETTINGS_OWNER, settingsForm, stageValues, syncSchema, useSettingsEditor } from "./settings-store"

function makeSchema(values: Record<string, number | boolean>): SettingsSchema {
  return {
    version: 1,
    categories: [
      {
        id: "c",
        label: "C",
        sections: [
          {
            id: "s",
            label: "S",
            items: [
              { type: "slider", name: "quality", label: "Quality", default: values.quality as number, minimum: 1, maximum: 100, step: 1, integer: true },
              { type: "toggle", name: "save", label: "Save", default: values.save as boolean, needs_restart: true },
              { type: "toggle", name: "previews", label: "Previews", default: values.previews as boolean, needs_reload_ui: true },
            ],
          },
        ],
      },
    ],
    quicksettings: [],
    values,
    defaults: { quality: 80, save: true, previews: true },
  }
}

const edit = (key: string, value: unknown) => settingsForm.getState().setValue(SETTINGS_OWNER, key, value)

describe("settings editor store", () => {
  beforeEach(() => {
    useSettingsEditor.setState({ schema: null, saved: {}, inputs: new Map(), applying: false, notice: { restart: [], reload: [] } })
    syncSchema(makeSchema({ quality: 90, save: true, previews: true }))
  })
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("loads the stored values and uses the built-in defaults for resets", () => {
    expect(editorValues()).toMatchObject({ quality: 90, save: true })
    expect(settingsForm.getState().defaults[SETTINGS_OWNER]).toMatchObject({ quality: 80 })
    settingsForm.getState().reset(SETTINGS_OWNER, ["quality"])
    expect(pendingChanges()).toEqual({ quality: 80 })
  })

  it("keeps unsaved edits (and drops those the server now stores) across a schema reload", () => {
    edit("quality", 50)
    edit("save", false)
    syncSchema(makeSchema({ quality: 70, save: false, previews: false }))
    expect(editorValues()).toMatchObject({ quality: 50, save: false, previews: false })
    expect(pendingChanges()).toEqual({ quality: 50 })
  })

  it("discards all or some edits; staged imports become edits", () => {
    stageValues({ quality: 10, save: false })
    discardChanges(["save"])
    expect(pendingChanges()).toEqual({ quality: 10 })
    discardChanges()
    expect(pendingChanges()).toEqual({})
  })

  it("applies optimistically and notes the settings that need a restart or reload", async () => {
    let resolve!: (value: SettingsSchema) => void
    const patch = vi.spyOn(api, "patchOptions").mockReturnValue(new Promise((r) => (resolve = r)))
    edit("save", false)
    edit("previews", false)
    const done = applyChanges()
    expect(patch).toHaveBeenCalledWith({ save: false, previews: false })
    // Shown as saved before the server answers.
    expect(pendingChanges()).toEqual({})
    expect(useSettingsEditor.getState().applying).toBe(true)
    resolve(makeSchema({ quality: 90, save: false, previews: false }))
    expect(await done).toEqual({ applied: ["save", "previews"], errors: [] })
    expect(useSettingsEditor.getState().notice).toEqual({ restart: ["save"], reload: ["previews"] })
    expect(queryClient.getQueryData<SettingsSchema>(keys.options)?.values.save).toBe(false)
  })

  it("rolls back rejected values and shows their messages", async () => {
    const body = { errors: [{ owner: "settings", param: "quality", message: "too high" }] }
    vi.spyOn(api, "patchOptions").mockRejectedValue(new ApiError(422, "quality: too high", body))
    edit("quality", 99)
    edit("save", false)
    const result = await applyChanges()
    expect(result.errors).toEqual(body.errors)
    // Both stay edits until the server's values arrive; the rejected one shows its message.
    expect(pendingChanges()).toEqual({ quality: 99, save: false })
    expect(settingsForm.getState().errors[SETTINGS_OWNER]).toEqual({ quality: "too high" })
    // The server applied `save` (an onchange failure elsewhere): the refetched schema settles it.
    syncSchema(makeSchema({ quality: 90, save: false, previews: true }))
    expect(pendingChanges()).toEqual({ quality: 99 })
    expect(settingsForm.getState().errors[SETTINGS_OWNER]).toEqual({ quality: "too high" })
  })

  it("keeps everything unsaved when the request fails outright", async () => {
    vi.spyOn(api, "patchOptions").mockRejectedValue(new ApiError(500, "boom"))
    edit("quality", 33)
    const result = await applyChanges()
    expect(result.applied).toEqual([])
    expect(pendingChanges()).toEqual({ quality: 33 })
  })
})
