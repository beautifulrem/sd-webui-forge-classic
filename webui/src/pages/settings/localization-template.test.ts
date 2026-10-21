import { describe, expect, it } from "vitest"
import type { SettingsSchema } from "@/ui/schema"
import { buildTemplate, settingsStrings } from "./localization-template"

describe("localization template", () => {
  it("collects the settings schema's strings", () => {
    const schema = {
      version: 1,
      categories: [
        {
          id: "c",
          label: "Saving",
          sections: [
            {
              id: "s",
              label: "Paths",
              items: [
                { type: "choice", name: "fmt", label: "Format", default: "png", notes: ["lossless"], links: [{ label: "wiki", url: "https://x" }], choices: [{ value: "png", label: "PNG" }] },
                { type: "row", children: [{ type: "toggle", name: "a", label: "Inner", default: true }] },
                { type: "html", text: "<b>Bold</b> help" },
                { type: "markdown", text: "**Bold** help" },
              ],
            },
          ],
        },
      ],
      quicksettings: [],
      values: {},
    } as unknown as SettingsSchema
    expect(settingsStrings(schema)).toEqual(["Saving", "Paths", "Format", "lossless", "wiki", "PNG", "Inner", "<b>Bold</b> help", "**Bold** help"])
  })

  it("sorts unique keys and fills known translations", () => {
    expect(buildTemplate(["b", "a", "b", " "], (key) => (key === "a" ? "甲" : undefined))).toEqual({ a: "甲", b: "" })
    expect(Object.keys(buildTemplate(["b", "a"], () => undefined))).toEqual(["a", "b"])
  })
})
