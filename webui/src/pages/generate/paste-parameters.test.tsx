import { renderHook } from "@testing-library/react"
import type { ReactNode } from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { InfotextParsed } from "@/api/types"
import { FormProvider } from "@/ui/context"
import { createFormStore } from "@/ui/form-store"

const mocks = vi.hoisted(() => ({
  api: { parseInfotext: vi.fn(), upload: vi.fn(), imageInfo: vi.fn() },
  readImageParameters: vi.fn(),
  applyPaste: vi.fn(),
  pastePlan: vi.fn(() => ({ entries: ["planned"] })),
  toast: { loading: vi.fn(() => "toast-1"), info: vi.fn(), success: vi.fn(), warning: vi.fn(), error: vi.fn() },
}))
vi.mock("@/api/endpoints", () => ({ api: mocks.api }))
vi.mock("@/features/infotext/read-image-info", () => ({ readImageParameters: mocks.readImageParameters }))
vi.mock("@/features/infotext/paste", () => ({ applyPaste: mocks.applyPaste, pastePlan: mocks.pastePlan }))
vi.mock("sonner", () => ({ toast: mocks.toast }))

const { usePasteParameters } = await import("./paste-parameters")

const store = createFormStore("paste-parameters-test", { persistDrafts: false })
const wrapper = ({ children }: { children: ReactNode }) => (
  <FormProvider formId="txt2img" store={store}>
    {children}
  </FormProvider>
)

const parsed = (problems: InfotextParsed["problems"] = []) => ({ core: { prompt: "a fox" }, scripts: {}, problems }) as unknown as InfotextParsed
const png = () => new File([new Uint8Array([137, 80, 78, 71])], "fox.png", { type: "image/png" })

function hook() {
  return renderHook(() => usePasteParameters(), { wrapper }).result.current
}

describe("usePasteParameters", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.applyPaste.mockReturnValue(3)
  })

  it("parses text on the server for this form and reports how many values it applied", async () => {
    mocks.api.parseInfotext.mockResolvedValue(parsed())
    await hook().fromText("a fox\nSteps: 20, Seed: 1, Size: 64x64")
    expect(mocks.api.parseInfotext).toHaveBeenCalledWith("a fox\nSteps: 20, Seed: 1, Size: 64x64", "txt2img")
    expect(mocks.applyPaste).toHaveBeenCalledWith({ store, tab: "txt2img" }, ["planned"])
    expect(mocks.toast.loading).toHaveBeenCalledWith("Reading parameters…")
    expect(mocks.toast.success).toHaveBeenCalledWith("Parameters applied: 3", { id: "toast-1" })
    expect(mocks.toast.warning).not.toHaveBeenCalled()
  })

  it("says so when nothing applied, and lists up to four skipped values", async () => {
    mocks.applyPaste.mockReturnValue(0)
    const problems = [1, 2, 3, 4, 5].map((n) => ({ param: n % 2 ? `param${n}` : undefined, owner: n % 2 ? undefined : `script${n}`, message: `bad ${n}` }))
    mocks.api.parseInfotext.mockResolvedValue(parsed(problems as InfotextParsed["problems"]))
    await hook().fromText("")
    expect(mocks.toast.info).toHaveBeenCalledWith("No parameters to apply", { id: "toast-1" })
    expect(mocks.toast.warning).toHaveBeenCalledWith("Some values were skipped", { description: "param1: bad 1\nscript2: bad 2\nparam3: bad 3\nscript4: bad 4" })
  })

  it("turns a failed request into an error toast", async () => {
    mocks.api.parseInfotext.mockRejectedValue(new Error("HTTP 500"))
    await hook().fromText("x")
    expect(mocks.toast.error).toHaveBeenCalledWith("Could not read the parameters", { id: "toast-1", description: "HTTP 500" })
    mocks.api.parseInfotext.mockRejectedValue("offline")
    await hook().fromText("x")
    expect(mocks.toast.error).toHaveBeenLastCalledWith("Could not read the parameters", { id: "toast-1", description: "offline" })
  })

  it("reads an image's parameters in the browser first", async () => {
    mocks.readImageParameters.mockResolvedValue("a fox\nSteps: 20, Seed: 1, Size: 64x64")
    mocks.api.parseInfotext.mockResolvedValue(parsed())
    await hook().fromImage(png())
    expect(mocks.api.parseInfotext).toHaveBeenCalledWith("a fox\nSteps: 20, Seed: 1, Size: 64x64", "txt2img")
    expect(mocks.api.upload).not.toHaveBeenCalled()
    expect(mocks.toast.success).toHaveBeenCalled()
  })

  it("lets the server read an image the browser cannot parse", async () => {
    mocks.readImageParameters.mockRejectedValue(new Error("corrupt EXIF"))
    mocks.api.upload.mockResolvedValue({ id: "upload-1" })
    mocks.api.imageInfo.mockResolvedValue({ parsed: parsed() })
    await hook().fromImage(png())
    expect(mocks.api.imageInfo).toHaveBeenCalledWith("upload-1", "txt2img")
    expect(mocks.applyPaste).toHaveBeenCalledOnce()
    expect(mocks.toast.success).toHaveBeenCalledWith("Parameters applied: 3", { id: "toast-1" })
  })

  it("tells when an image has no parameters at all, and when reading it failed", async () => {
    mocks.readImageParameters.mockResolvedValue(null)
    mocks.api.upload.mockResolvedValue({ id: "upload-2" })
    mocks.api.imageInfo.mockResolvedValue({ parsed: null })
    await hook().fromImage(png())
    expect(mocks.toast.info).toHaveBeenCalledWith("This image has no generation parameters", { id: "toast-1" })
    expect(mocks.applyPaste).not.toHaveBeenCalled()

    mocks.api.upload.mockRejectedValue(new Error("The file is too large."))
    await hook().fromImage(png())
    expect(mocks.toast.error).toHaveBeenCalledWith("Could not read the parameters", { id: "toast-1", description: "The file is too large." })
  })
})
