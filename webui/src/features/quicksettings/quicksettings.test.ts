import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import { bootstrap } from "@/api/mock/fixtures"
import { keys, queryClient } from "@/api/queries"
import type { Bootstrap } from "@/api/types"
import type { SettingsSchema } from "@/ui/schema"
import { setPinned, quickKeys, withPin } from "./pinned"
import { createQuickForm, QUICK_OWNER } from "./quick-form"

const schema: SettingsSchema = {
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
            { type: "toggle", name: "samples_save", label: "Save", default: true },
            { type: "row", children: [{ type: "slider", name: "jpeg_quality", label: "Quality", default: 80, minimum: 1, maximum: 100, step: 1, integer: true }] },
          ],
        },
      ],
    },
  ],
  quicksettings: ["samples_save", "jpeg_quality"],
  values: { samples_save: true, jpeg_quality: 80 },
  defaults: { samples_save: true, jpeg_quality: 80 },
}

describe("pinned keys", () => {
  it("skips the model bar's settings, duplicates and junk", () => {
    expect(quickKeys(["sd_model_checkpoint", "jpeg_quality", "jpeg_quality", 3, "forge_preset", "samples_save"])).toEqual(["jpeg_quality", "samples_save"])
    expect(quickKeys(null)).toEqual([])
  })

  it("adds at the end and removes", () => {
    expect(withPin(["a", "b"], "a", true)).toEqual(["b", "a"])
    expect(withPin(["a", "b"], "a", false)).toEqual(["b"])
  })

  describe("writes", () => {
    beforeEach(() => queryClient.setQueryData(keys.app, { ...structuredClone(bootstrap), options: { ...bootstrap.options, quicksettings_list: ["sd_model_checkpoint"] } }))
    afterEach(() => {
      vi.restoreAllMocks()
      queryClient.clear()
    })

    it("pins optimistically", async () => {
      const patch = vi.spyOn(api, "patchOptions").mockResolvedValue(schema)
      const done = setPinned("jpeg_quality", true)
      expect(queryClient.getQueryData<Bootstrap>(keys.app)?.options.quicksettings_list).toEqual(["sd_model_checkpoint", "jpeg_quality"])
      await done
      expect(patch).toHaveBeenCalledWith({ quicksettings_list: ["sd_model_checkpoint", "jpeg_quality"] })
    })

    it("rolls back when the server refuses", async () => {
      vi.spyOn(api, "patchOptions").mockRejectedValue(new ApiError(422, "frozen"))
      vi.spyOn(api, "app").mockReturnValue(new Promise(() => {}))
      await setPinned("jpeg_quality", true)
      expect(queryClient.getQueryData<Bootstrap>(keys.app)?.options.quicksettings_list).toEqual(["sd_model_checkpoint"])
    })
  })
})

describe("quick settings form", () => {
  afterEach(() => vi.useRealTimers())

  it("writes discrete changes at once and continuous ones after a pause", async () => {
    vi.useFakeTimers()
    const write = vi.fn(async () => ({}))
    const form = createQuickForm(write, 400)
    const stop = form.start()
    form.load(schema, ["jpeg_quality", "samples_save", "missing"])
    expect(form.nodes.map((node) => node.name)).toEqual(["jpeg_quality", "samples_save"])
    expect(write).not.toHaveBeenCalled()

    form.store.getState().setValue(QUICK_OWNER, "samples_save", false)
    await vi.waitFor(() => expect(write).toHaveBeenCalledWith({ samples_save: false }))

    form.store.getState().setValue(QUICK_OWNER, "jpeg_quality", 60)
    form.store.getState().setValue(QUICK_OWNER, "jpeg_quality", 70)
    expect(write).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(400)
    expect(write).toHaveBeenLastCalledWith({ jpeg_quality: 70 })
    expect(write).toHaveBeenCalledTimes(2)
    stop()
  })

  it("reverts a refused value and shows the message", async () => {
    const body = { errors: [{ owner: "settings", param: "samples_save", message: "frozen" }] }
    const form = createQuickForm(async () => Promise.reject(new ApiError(422, "frozen", body)))
    const stop = form.start()
    form.load(schema, ["samples_save"])
    form.store.getState().setValue(QUICK_OWNER, "samples_save", false)
    await vi.waitFor(() => expect(form.store.getState().values[QUICK_OWNER]?.samples_save).toBe(true))
    expect(form.store.getState().errors[QUICK_OWNER]).toEqual({ samples_save: "frozen" })
    stop()
  })

  it("flushes a pending write when stopped", async () => {
    vi.useFakeTimers()
    const write = vi.fn(async () => ({}))
    const form = createQuickForm(write, 400)
    const stop = form.start()
    form.load(schema, ["jpeg_quality"])
    form.store.getState().setValue(QUICK_OWNER, "jpeg_quality", 42)
    stop()
    expect(write).toHaveBeenCalledWith({ jpeg_quality: 42 })
  })
})
