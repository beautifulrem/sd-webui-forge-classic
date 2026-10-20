/** The result actions without UI: their requests and their progress / result toasts. */
import { toast } from "sonner"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { recordFetch } from "@/test/fetch-recorder"
import { getFormStore } from "@/ui/form-store"
import { tabOwners } from "@/ui/tab"
import { applyParameters, findByInfotext, reuseSeed, saveResults, sendWithFeedback } from "./result-actions"
import { sendTo } from "./send-to"

vi.mock("./send-to", async (original) => ({ ...(await original<typeof import("./send-to")>()), sendTo: vi.fn() }))

const sent = vi.mocked(sendTo)
let loading: ReturnType<typeof vi.spyOn>
let success: ReturnType<typeof vi.spyOn>
let failure: ReturnType<typeof vi.spyOn>

beforeEach(() => {
  loading = vi.spyOn(toast, "loading").mockReturnValue("toast-1")
  success = vi.spyOn(toast, "success")
  failure = vi.spyOn(toast, "error")
  getFormStore("txt2img").getState().load(tabOwners(tabSchema("txt2img")))
})
afterEach(() => {
  vi.restoreAllMocks()
  sent.mockReset()
})

describe("finding inputs by infotext key", () => {
  it("returns the owner and name of the input that reads a key, or null", () => {
    const index = getFormStore("txt2img").getState().index
    expect(findByInfotext(index, "Seed")).toEqual(["seed", "seed"])
    expect(findByInfotext(index, "No Such Key")).toBeNull()
  })

  it("does not reuse a seed when the form has no seed input", () => {
    getFormStore("txt2img").getState().load([])
    expect(reuseSeed("txt2img", "x\nSteps: 20, Seed: 7")).toBe(false)
  })
})

describe("using a result's parameters", () => {
  it("pastes them into the form and reports how many were applied", async () => {
    sent.mockResolvedValue(5)
    applyParameters("img2img", "a cat\nSteps: 12")
    expect(sent).toHaveBeenCalledWith("img2img", { infotext: "a cat\nSteps: 12" }, {})
    expect(loading).toHaveBeenCalledWith("Reading parameters…")
    await vi.waitFor(() => expect(success).toHaveBeenCalledWith("Parameters applied: 5", { id: "toast-1" }))
  })

  it("replaces the progress toast with the reason when they cannot be read", async () => {
    sent.mockRejectedValue(new Error("not a generation"))
    applyParameters("txt2img", "garbage")
    await vi.waitFor(() => expect(failure).toHaveBeenCalledWith("Could not read the parameters", { id: "toast-1", description: "not a generation" }))
    sent.mockRejectedValue("offline")
    applyParameters("txt2img", "garbage")
    await vi.waitFor(() => expect(failure).toHaveBeenLastCalledWith("Could not read the parameters", { id: "toast-1", description: "offline" }))
  })
})

describe("sending a result", () => {
  it("names the target in the progress and result toasts", async () => {
    sent.mockResolvedValue(3)
    sendWithFeedback("inpaint", { fileId: "f1" })
    expect(sent).toHaveBeenCalledWith("inpaint", { fileId: "f1" })
    expect(loading).toHaveBeenCalledWith("Sending to Inpaint…")
    await vi.waitFor(() => expect(success).toHaveBeenCalledWith("Sent to Inpaint", { id: "toast-1" }))
  })

  it("reports a failed send with its reason", async () => {
    sent.mockRejectedValue(new Error("the image was deleted"))
    sendWithFeedback("extras", { fileId: "gone" })
    await vi.waitFor(() => expect(failure).toHaveBeenCalledWith("Could not send to Extras", { id: "toast-1", description: "the image was deleted" }))
  })
})

describe("saving results", () => {
  it("copies outputs to the Save folder", async () => {
    const fetch = recordFetch(() => ({ body: { saved: ["a.png", "b.png"], zip: null } }))
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {})
    await saveResults(["f1", "f2"])
    expect(fetch.last()).toMatchObject({ method: "POST", path: "/api/v2/outputs/save", body: { ids: ["f1", "f2"], zip: false } })
    expect(success).toHaveBeenCalledWith("Saved: 2", { id: "toast-1" })
    expect(click).not.toHaveBeenCalled()
  })

  it("downloads the zip of a generation", async () => {
    recordFetch(() => ({ body: { saved: ["a.png"], zip: "zip-1" } }))
    let href = ""
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
      href = this.getAttribute("href") ?? ""
    })
    await saveResults(["f1"], true)
    expect(href).toBe("/api/v2/files/zip-1?download=1")
    expect(success).toHaveBeenCalledWith("Saved: 1", { id: "toast-1" })
  })

  it("explains a failed save", async () => {
    recordFetch(() => ({ status: 403, body: { detail: "saving is disabled" } }))
    await saveResults(["f1"])
    expect(failure).toHaveBeenCalledWith("Could not save", { id: "toast-1", description: "saving is disabled" })
    expect(success).not.toHaveBeenCalled()
  })
})
