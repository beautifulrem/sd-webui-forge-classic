/** The toolbar of one result (stage, lightbox): which actions it offers and what each one does. */
import { QueryClientProvider } from "@tanstack/react-query"
import { configure, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { toast } from "sonner"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { keys, queryClient } from "@/api/queries"
import type { Bootstrap } from "@/api/types"
import { type ResultItem, useSession } from "@/features/gallery/session-store"
import { upscaleResult } from "@/features/progress/run-controller"
import { bootstrapWith } from "@/test/api-fixtures"
import { getFormStore } from "@/ui/form-store"
import { tabOwners } from "@/ui/tab"
import { applyParameters, saveResults, sendWithFeedback } from "./result-actions"
import { ResultActions } from "./ResultActions"

vi.mock("./result-actions", async (original) => ({
  ...(await original<typeof import("./result-actions")>()),
  applyParameters: vi.fn(),
  saveResults: vi.fn(async () => {}),
  sendWithFeedback: vi.fn(),
}))
vi.mock("@/features/progress/run-controller", async (original) => ({
  ...(await original<typeof import("@/features/progress/run-controller")>()),
  upscaleResult: vi.fn(),
}))

configure({ asyncUtilTimeout: 5_000 })

const INFOTEXT = "a cat\nSteps: 20, Seed: 4242, Size: 512x512"
const result = (extra: Partial<ResultItem> = {}): ResultItem => ({
  key: "j1:0",
  fileId: "f1",
  jobId: "j1",
  tab: "txt2img",
  index: 0,
  infotext: INFOTEXT,
  createdAt: 1,
  ...extra,
})

function show(props: Partial<Parameters<typeof ResultActions>[0]> = {}) {
  const onSent = vi.fn()
  render(
    <QueryClientProvider client={queryClient}>
      <ResultActions item={result()} tab="txt2img" onSent={onSent} {...props} />
    </QueryClientProvider>,
  )
  return { onSent, toolbar: screen.getByRole("toolbar", { name: "Image actions" }) }
}

const button = (name: string) => screen.getByRole("button", { name })

/** Opens a dropdown of the toolbar and picks an entry. */
async function pick(trigger: string, entry: RegExp) {
  fireEvent.click(button(trigger))
  const menu = await screen.findByRole("menu")
  fireEvent.click(within(menu).getByRole("menuitem", { name: entry }))
}

beforeEach(() => {
  getFormStore("txt2img").getState().load(tabOwners(tabSchema("txt2img")))
  queryClient.setQueryData<Bootstrap>(keys.app, bootstrapWith({}))
})
afterEach(() => {
  vi.restoreAllMocks()
  vi.mocked(sendWithFeedback).mockReset()
  vi.mocked(applyParameters).mockReset()
  vi.mocked(saveResults).mockClear()
  vi.mocked(upscaleResult).mockReset()
  useSession.getState().clear()
  queryClient.clear()
})

describe("result actions", { timeout: 20_000 }, () => {
  it("disables everything that needs a result when there is none", () => {
    show({ item: null })
    for (const name of ["Download", "Save", "Open in new tab", "Copy generation parameters", "Reuse seed", "Use these parameters", "Remove from history"]) {
      expect(button(name)).toBeDisabled()
    }
    expect(screen.getByRole("button", { name: /Send to/ })).toBeDisabled()
  })

  it("downloads and opens the output file", () => {
    let href = ""
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
      href = this.getAttribute("href") ?? ""
    })
    const open = vi.spyOn(window, "open").mockReturnValue(null)
    show()
    fireEvent.click(button("Download"))
    expect(href).toBe("/api/v2/files/f1?download=1")
    fireEvent.click(button("Open in new tab"))
    expect(open).toHaveBeenCalledWith("/api/v2/files/f1", "_blank", "noopener,noreferrer")
  })

  it("copies the parameters, and says so when the clipboard refuses", async () => {
    const writeText = vi.fn().mockResolvedValueOnce(undefined).mockRejectedValueOnce(new Error("denied"))
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true })
    const success = vi.spyOn(toast, "success")
    const failure = vi.spyOn(toast, "error")
    show()
    fireEvent.click(button("Copy generation parameters"))
    await waitFor(() => expect(success).toHaveBeenCalledWith("Copied"))
    expect(writeText).toHaveBeenCalledWith(INFOTEXT)
    fireEvent.click(button("Copy generation parameters"))
    await waitFor(() => expect(failure).toHaveBeenCalledWith("Could not copy"))
  })

  it("reuses the seed in the form, or explains that the image has none", () => {
    const success = vi.spyOn(toast, "success")
    const failure = vi.spyOn(toast, "error")
    show()
    fireEvent.click(button("Reuse seed"))
    expect(getFormStore("txt2img").getState().values.seed?.seed).toBe(4242)
    expect(success).toHaveBeenCalledWith("Seed reused")
    getFormStore("txt2img").getState().load([])
    fireEvent.click(button("Reuse seed"))
    expect(failure).toHaveBeenCalledWith("This image has no seed to reuse")
  })

  it("uses the parameters in the toolbar's form, never in Extras, and not without infotext", () => {
    show({ tab: "img2img" })
    fireEvent.click(button("Use these parameters"))
    expect(applyParameters).toHaveBeenCalledWith("img2img", INFOTEXT)
  })

  it("has nothing to paste for Extras results or images without parameters", () => {
    show({ tab: "extras", item: result({ infotext: null }) })
    expect(button("Use these parameters")).toBeDisabled()
    expect(button("Copy generation parameters")).toBeDisabled()
    expect(button("Reuse seed")).toBeDisabled()
  })

  it("saves the result, or its whole generation as a zip", async () => {
    useSession.getState().addJob({ jobId: "j1", tab: "txt2img", outputs: [{ id: "f1" }, { id: "f2" }] as never, infotext: INFOTEXT })
    useSession.getState().addJob({ jobId: "j2", tab: "txt2img", outputs: [{ id: "other" }] as never, infotext: INFOTEXT })
    show()
    await pick("Save", /Save to the Save folder/)
    expect(saveResults).toHaveBeenLastCalledWith(["f1"])
    await pick("Save", /Save this generation as zip/)
    expect(saveResults).toHaveBeenLastCalledWith(["f1", "f2"], true)
  })

  it("sends a txt2img result with its parameters and tells the lightbox it was sent", async () => {
    const { onSent } = show()
    await pick("Send to", /^img2img$/)
    expect(sendWithFeedback).toHaveBeenLastCalledWith("img2img", result())
    expect(onSent).toHaveBeenCalledTimes(1)
    await pick("Send to", /Inpaint/)
    expect(sendWithFeedback).toHaveBeenLastCalledWith("inpaint", result())
    await pick("Send to", /Extras/)
    expect(sendWithFeedback).toHaveBeenLastCalledWith("extras", result())
    await pick("Send to", /txt2img \(parameters\)/)
    expect(sendWithFeedback).toHaveBeenLastCalledWith("txt2img", { infotext: INFOTEXT })
    expect(onSent).toHaveBeenCalledTimes(4)
  })

  it("sends only the image of an img2img result unless allow_i2i_send_info is on (A1111)", async () => {
    show({ tab: "img2img", item: result({ tab: "img2img" }) })
    await pick("Send to", /^img2img$/)
    expect(sendWithFeedback).toHaveBeenLastCalledWith("img2img", { fileId: "f1" })
    queryClient.setQueryData<Bootstrap>(keys.app, bootstrapWith({ allow_i2i_send_info: true }))
    await pick("Send to", /Inpaint/)
    expect(sendWithFeedback).toHaveBeenLastCalledWith("inpaint", result({ tab: "img2img" }))
  })

  it("offers Hires. fix only for first-pass txt2img images", () => {
    show({ upscalable: true })
    fireEvent.click(button("Upscale with Hires. fix"))
    expect(upscaleResult).toHaveBeenCalledWith({ fileId: "f1", key: "j1:0" })
  })

  it("does not upscale grids, extra images or video frames, and hides the button elsewhere", () => {
    show({ upscalable: true, item: result({ role: "grid" }) })
    expect(button("Upscale with Hires. fix")).toBeDisabled()
    render(<ResultActions item={result({ tab: "img2img" })} tab="img2img" upscalable />)
    expect(screen.getAllByRole("button", { name: "Upscale with Hires. fix" })).toHaveLength(1)
  })

  it("removes the result from the session history unless it is not removable", () => {
    useSession.getState().addJob({ jobId: "j1", tab: "txt2img", outputs: [{ id: "f1" }] as never, infotext: INFOTEXT })
    show()
    fireEvent.click(button("Remove from history"))
    expect(useSession.getState().items).toEqual([])
    render(<ResultActions item={result()} tab="txt2img" removable={false} />)
    expect(screen.getAllByRole("button", { name: "Remove from history" })).toHaveLength(1)
  })
})
