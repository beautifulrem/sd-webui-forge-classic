import { describe, expect, it } from "vitest"
import { ApiError } from "@/api/http"
import { errorText, validationText } from "./error-text"

const failed = (status: number) => new ApiError(status, `HTTP ${status}`, { detail: "English detail for the logs" })

describe("errorText", () => {
  it("says the server cannot be reached for anything but an API error", () => {
    expect(errorText(new TypeError("Failed to fetch"))).toBe("The server cannot be reached.")
    expect(errorText("offline")).toBe("The server cannot be reached.")
    expect(errorText(undefined)).toBe("The server cannot be reached.")
  })

  it.each([
    [401, "Sign in again to continue."],
    [403, "The server does not allow this."],
    [404, "It is no longer there."],
    [409, "It changed meanwhile; refresh and try again."],
    [413, "The file is too large."],
    [415, "This file cannot be read as an image."],
    [422, "The server rejected the request."],
    [503, "The server is busy; try again shortly."],
  ])("explains status %i in a sentence, not the server's English detail", (status, text) => {
    expect(errorText(failed(status))).toBe(text)
  })

  it("falls back on the status class for other codes", () => {
    expect(errorText(failed(500))).toBe("The server ran into an error (500).")
    expect(errorText(failed(502))).toBe("The server ran into an error (502).")
    expect(errorText(failed(400))).toBe("The request failed (400).")
    expect(errorText(failed(429))).toBe("The request failed (429).")
  })

  it("prefers the caller's own sentence for a status", () => {
    const texts = { 404: "The image was deleted.", 409: "Someone else saved first." }
    expect(errorText(failed(404), texts)).toBe("The image was deleted.")
    expect(errorText(failed(409), texts)).toBe("Someone else saved first.")
    // Statuses the caller did not name keep the generic text.
    expect(errorText(failed(403), texts)).toBe("The server does not allow this.")
    expect(errorText(new Error("network"), texts)).toBe("The server cannot be reached.")
  })
})

describe("validationText", () => {
  it("lists each distinct message once, translated like the inline errors", async () => {
    const { i18n } = await import("@/i18n")
    i18n.addResource("zh-CN", "server", "an image is required", "需要一张图像")
    i18n.addResource("zh-CN", "server", "paint a mask over the image", "请在图像上绘制蒙版")
    const errors = [
      { owner: "core", param: "init_img", message: "an image is required" },
      { owner: "core", param: "mask", message: "paint a mask over the image" },
      { owner: "other", param: "image", message: "an image is required" },
    ]
    expect(validationText(errors)).toBe("an image is required · paint a mask over the image")
    await i18n.changeLanguage("zh-CN")
    try {
      expect(validationText(errors)).toBe("需要一张图像 · 请在图像上绘制蒙版")
    } finally {
      await i18n.changeLanguage("en")
    }
  })
})
