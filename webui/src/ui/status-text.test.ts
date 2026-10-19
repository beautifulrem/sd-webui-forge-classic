import { describe, expect, it } from "vitest"
import { parseStatus } from "./status-text"

describe("parseStatus", () => {
  it("turns a leading status emoji into a tone", () => {
    expect(parseStatus("✅ Model loaded (bf16): ready")).toEqual({ tone: "success", text: "Model loaded (bf16): ready" })
    expect(parseStatus("⚪ 尚未下载权重")).toEqual({ tone: "info", text: "尚未下载权重" })
    expect(parseStatus("⚠️ Not set — Gelbooru grabs return 401.")).toEqual({ tone: "warning", text: "Not set — Gelbooru grabs return 401." })
    expect(parseStatus("❌ Failed to unload: boom")).toEqual({ tone: "error", text: "Failed to unload: boom" })
    expect(parseStatus("✓ Rule34 credentials set.")).toEqual({ tone: "success", text: "Rule34 credentials set." })
  })

  it("leaves other text alone", () => {
    expect(parseStatus("Captioned 3 images")).toEqual({ tone: null, text: "Captioned 3 images" })
    expect(parseStatus("")).toEqual({ tone: null, text: "" })
  })
})
