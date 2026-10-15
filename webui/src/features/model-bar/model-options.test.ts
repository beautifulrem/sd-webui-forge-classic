import { describe, expect, it } from "vitest"
import { checkpointLabel, fileName, formatBytes, presetLabel } from "./model-options"

describe("model bar helpers", () => {
  it("formats sizes", () => {
    expect(formatBytes(6.46 * 1024 ** 3)).toBe("6.5 GB")
    expect(formatBytes(317 * 1024 ** 2)).toBe("317 MB")
    expect(formatBytes(null)).toBeNull()
  })

  it("drops the hash from checkpoint titles and paths from module values", () => {
    expect(checkpointLabel("anima-turbo-v1.1.safetensors [5e1f00aa]")).toBe("anima-turbo-v1.1.safetensors")
    expect(checkpointLabel("plain.safetensors")).toBe("plain.safetensors")
    expect(fileName("/models/VAE/qwen_image_vae.safetensors")).toBe("qwen_image_vae.safetensors")
    expect(fileName("C:\\models\\text_encoder\\qwen.safetensors")).toBe("qwen.safetensors")
  })

  it("names presets", () => {
    expect(presetLabel("xl")).toBe("SDXL")
    expect(presetLabel("future")).toBe("future")
  })
})
