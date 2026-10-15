import { describe, expect, it } from "vitest"
import { i18n } from "@/i18n"
import { registerWidgetCatalogs, widgetKeys } from "./i18n"

describe("widget catalogs", () => {
  it("register under the widget's namespace, by the file's locale", () => {
    const ns = registerWidgetCatalogs("probe", { "./locales/zh-CN.json": { "Hello {{name}}": "你好 {{name}}" } })
    expect(ns).toBe("widget:probe")
    expect(i18n.getFixedT("zh-CN", ns)("Hello {{name}}", { name: "Remi" })).toBe("你好 Remi")
    expect(i18n.getFixedT("en", ns)("Hello {{name}}", { name: "Remi" })).toBe("Hello Remi")
  })

  it("finds the literal keys a source passes to tw()", () => {
    expect(widgetKeys('tw("One") + t("Two") + tw( "Three \\"q\\"", {})')).toEqual(["One", 'Three \\"q\\"'])
  })
})
