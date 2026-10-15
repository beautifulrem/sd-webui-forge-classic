import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { api } from "@/api/endpoints"
import { keys, queryClient } from "@/api/queries"
import type { Bootstrap } from "@/api/types"
import { navigate } from "@/app/router"
import { getFormStore } from "@/ui/form-store"
import { tabOwners } from "@/ui/tab"
import { reuseSeed } from "./result-actions"
import { deliver, formOf, imageEntry, sendTo } from "./send-to"
import { bootstrapWith } from "@/test/api-fixtures"

const load = (tab: "txt2img" | "img2img") => getFormStore(tab).getState().load(tabOwners(tabSchema(tab)))

function setOptions(options: Record<string, unknown>) {
  queryClient.setQueryData<Bootstrap>(keys.app, bootstrapWith(options))
}

beforeEach(() => {
  navigate({ name: "generate", tab: "txt2img" }, { replace: true })
  load("txt2img")
})
afterEach(() => {
  vi.restoreAllMocks()
  queryClient.removeQueries({ queryKey: keys.app })
})

describe("send-to mapping", () => {
  it("maps targets to forms and image params", () => {
    expect(formOf("inpaint")).toBe("img2img")
    expect(imageEntry("img2img", "u1")).toEqual({ owner: "core", values: { init_img: "u1", mode: "img2img" } })
    expect(imageEntry("inpaint", "u1")).toEqual({ owner: "core", values: { init_img: "u1", mode: "Inpaint" } })
    expect(imageEntry("extras", "u1")).toEqual({ owner: "core", values: { image: "u1", mode: "Single Image" } })
    expect(imageEntry("txt2img", "u1")).toBeNull()
  })
})

describe("deliver", () => {
  it("clears the mask and sketch of the previous source image", async () => {
    load("img2img")
    const store = getFormStore("img2img")
    store.getState().setValues("core", { init_img: "old", mask: "old-mask" })
    const count = await deliver("img2img", [imageEntry("inpaint", "new")!])
    // init_img, mode and the two layers painted over it (the uploaded mask is not).
    expect(count).toBe(4)
    expect(store.getState().values.core).toMatchObject({ init_img: "new", mode: "Inpaint", mask: null, sketch: null })
    // Layers sent along with the image are kept.
    await deliver("img2img", [{ owner: "core", values: { init_img: "newer", mask: "its-mask" } }])
    expect(store.getState().values.core).toMatchObject({ init_img: "newer", mask: "its-mask" })
  })

  it("applies at once to the form on screen", async () => {
    const count = await deliver("txt2img", [{ owner: "core", values: { prompt: "now" } }])
    expect(count).toBe(1)
    expect(getFormStore("txt2img").getState().values.core?.prompt).toBe("now")
  })

  it("applies to a loaded generate form at once and shows it", async () => {
    load("img2img")
    expect(await deliver("img2img", [{ owner: "core", values: { prompt: "kept" } }])).toBe(1)
    expect(location.hash).toBe("#/img2img")
    expect(getFormStore("img2img").getState().values.core?.prompt).toBe("kept")
  })

  it("waits for a form's first schema load, then applies (upload ids included)", async () => {
    getFormStore("img2img").setState({ revision: 0 })
    const pending = deliver("img2img", [{ owner: "core", values: { init_img: "upload-9", mode: "Inpaint", prompt: "later" } }])
    expect(location.hash).toBe("#/img2img")
    load("img2img")
    // init_img, mode, prompt and the cleared mask and sketch.
    expect(await pending).toBe(5)
    expect(getFormStore("img2img").getState().values.core).toMatchObject({ init_img: "upload-9", mode: "Inpaint", prompt: "later" })
  })
})

describe("sendTo", () => {
  it("sends the output file as the source image, pastes the parameters, respects send_seed / send_size", async () => {
    setOptions({ send_seed: false, send_size: true })
    const parse = vi.spyOn(api, "parseInfotext").mockResolvedValue({
      tab: "img2img",
      core: { prompt: "a cat", width: 640, height: 768 },
      scripts: { seed: { seed: 42 }, sampler: { steps: 12 } },
    })
    getFormStore("img2img").setState({ revision: 0 })
    const sent = sendTo("img2img", { fileId: "out-1", infotext: "a cat\nSteps: 12, Seed: 42, Size: 640x768" })
    await vi.waitFor(() => expect(location.hash).toBe("#/img2img"))
    load("img2img")
    await sent
    expect(parse).toHaveBeenCalledWith("a cat\nSteps: 12, Seed: 42, Size: 640x768", "img2img")
    const values = getFormStore("img2img").getState().values
    expect(values.core).toMatchObject({ init_img: "out-1", mode: "img2img", prompt: "a cat", width: 640, height: 768 })
    expect(values.sampler?.steps).toBe(12)
    expect(values.seed?.seed).toBe(-1) // send_seed off
  })
})

describe("reuseSeed", () => {
  it("sets the form's seed param from the infotext", () => {
    expect(reuseSeed("txt2img", "x\nSteps: 20, Seed: 1234567, Size: 512x512")).toBe(true)
    expect(getFormStore("txt2img").getState().values.seed?.seed).toBe(1234567)
    expect(reuseSeed("txt2img", "no parameters")).toBe(false)
  })
})
