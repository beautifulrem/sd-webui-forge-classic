import { describe, expect, it } from "vitest"
import { formatShortcut } from "@/app/commands"
import { i18n } from "@/i18n"
import { placeholderText } from "./placeholder"

describe("placeholderText", () => {
  it("names the generate shortcuts the way the Generate button does", () => {
    const text = placeholderText(i18n.t, "Prompt\n({{generate}} to Generate ; {{skip}} to Skip ; {{interrupt}} to Interrupt)")
    expect(text).toBe(`Prompt\n(${formatShortcut("Mod+Enter")} to Generate ; ${formatShortcut("Alt+Enter")} to Skip ; Esc to Interrupt)`)
  })

  it("keeps other placeholders and nothing as they are", () => {
    expect(placeholderText(i18n.t, "Path to a video file")).toBe("Path to a video file")
    expect(placeholderText(i18n.t, null)).toBeUndefined()
  })
})
