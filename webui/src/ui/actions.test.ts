import { describe, expect, it } from "vitest"
import { splitResponse } from "./actions"

describe("splitResponse", () => {
  it("separates reserved keys and resolves cross-owner params", () => {
    const split = splitResponse("anima", {
      preset: { value: "detail", label: "Preset", bogus: 1 },
      "core.steps": { value: 40, maximum: 80 },
      __toast__: { level: "success", message: "ok" },
      __outputs__: { status: "done" },
      __settings__: { sd_vae: "Automatic" },
      __future__: { anything: true },
      broken: "not an object",
    })
    expect(split.params).toEqual([
      ["anima", "preset", { value: "detail", label: "Preset" }],
      ["core", "steps", { value: 40, maximum: 80 }],
    ])
    expect(split.toast).toEqual({ level: "success", message: "ok" })
    expect(split.outputs).toEqual({ status: "done" })
    expect(split.settings).toEqual({ sd_vae: "Automatic" })
  })

  it("takes a toast message with params (remi.ui.message)", () => {
    const message = { text: "Saved template: {{name}}", params: { name: "mine" } }
    expect(splitResponse("x", { __toast__: { level: "success", message } }).toast).toEqual({ level: "success", message })
    expect(splitResponse("x", { __toast__: { message: { text: 3 } } }).toast).toBeUndefined()
  })

  it("tolerates empty and malformed responses", () => {
    expect(splitResponse("x", null)).toEqual({ params: [] })
    expect(splitResponse("x", { __toast__: { message: 3 } })).toEqual({ params: [] })
  })
})
