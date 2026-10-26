import { afterEach, describe, expect, it } from "vitest"
import { spaceInterpolations } from "./cjk-spacing"
import { i18n } from "./index"

describe("spaceInterpolations", () => {
  it("spaces a Latin value from the Han text around it", () => {
    expect(spaceInterpolations("启用ADetailer", ["ADetailer"])).toBe("启用 ADetailer")
    expect(spaceInterpolations("启用Sparse Attention 集成", ["Sparse Attention 集成"])).toBe("启用 Sparse Attention 集成")
    expect(spaceInterpolations("LoRA的标签", ["LoRA"])).toBe("LoRA 的标签")
    expect(spaceInterpolations("第3步", ["3"])).toBe("第 3 步")
  })

  it("leaves Han next to Han, existing spaces, punctuation and the text around values alone", () => {
    expect(spaceInterpolations("启用高分辨率修复", ["高分辨率修复"])).toBe("启用高分辨率修复")
    expect(spaceInterpolations("启用 ADetailer", ["ADetailer"])).toBe("启用 ADetailer")
    expect(spaceInterpolations("已发送到「Extras」", ["Extras"])).toBe("已发送到「Extras」")
    expect(spaceInterpolations("VAE模型：ADetailer", ["ADetailer"])).toBe("VAE模型：ADetailer")
    expect(spaceInterpolations("", ["x"])).toBe("")
  })

  it("handles every occurrence and values that repeat", () => {
    expect(spaceInterpolations("A和A", ["A", "A"])).toBe("A 和 A")
  })
})

describe("the i18n instance", () => {
  afterEach(async () => {
    await i18n.changeLanguage("en")
  })

  it("spaces interpolated values in Chinese only", async () => {
    i18n.addResourceBundle("zh-CN", "chrome", { "Enable {{title}}": "启用{{title}}" }, true, true)
    await i18n.changeLanguage("zh-CN")
    expect(i18n.t("Enable {{title}}", { title: "ADetailer" })).toBe("启用 ADetailer")
    expect(i18n.t("Enable {{title}}", { title: "高分辨率修复" })).toBe("启用高分辨率修复")
    await i18n.changeLanguage("en")
    expect(i18n.t("Enable {{title}}", { title: "ADetailer" })).toBe("Enable ADetailer")
  })
})
