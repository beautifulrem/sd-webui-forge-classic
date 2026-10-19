import { beforeEach, describe, expect, it, vi } from "vitest"

const helpers = vi.hoisted(() => ({ resetForm: vi.fn(), resetLosesWork: vi.fn() }))
vi.mock("./run-helpers", () => helpers)

const { requestReset, useResetRequest } = await import("./reset-request")

describe("requestReset", () => {
  beforeEach(() => {
    helpers.resetForm.mockReset()
    helpers.resetLosesWork.mockReset()
    useResetRequest.setState({ tab: null })
  })

  it("resets at once when nothing would be lost", () => {
    helpers.resetLosesWork.mockReturnValue(false)
    requestReset("txt2img")
    expect(helpers.resetLosesWork).toHaveBeenCalledWith("txt2img")
    expect(helpers.resetForm).toHaveBeenCalledWith("txt2img")
    expect(useResetRequest.getState().tab).toBeNull()
  })

  it("asks first when prompts or images would be lost", () => {
    helpers.resetLosesWork.mockReturnValue(true)
    requestReset("img2img")
    expect(helpers.resetForm).not.toHaveBeenCalled()
    expect(useResetRequest.getState().tab).toBe("img2img")
  })
})
