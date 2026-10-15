import { describe, expect, it } from "vitest"
import img2imgContract from "@/api/mock/contract/img2img.json"
import txt2imgContract from "@/api/mock/contract/txt2img.json"
import { tabSchema } from "@/api/mock/fixtures"
import { inputsIn } from "@/ui/form-index"
import type { Node, TabSchema } from "@/ui/schema"
import { curateCore } from "./curate"

const names = (node: Node) => inputsIn(node).map((input) => input.name)

describe("curateCore", () => {
  it("arranges the real txt2img form", () => {
    const curated = curateCore((txt2imgContract as unknown as TabSchema).core.layout)
    expect(curated.prompt?.name).toBe("prompt")
    expect(curated.negative?.name).toBe("negative_prompt")
    expect(curated.styles?.name).toBe("styles")
    expect(curated.script?.name).toBe("script")
    // The server's category order (ui_reorder_list) is kept.
    expect(curated.blocks.map((b) => b.kind)).toEqual(["sampler", "sections", "other", "size", "guidance", "seed", "other", "other"])
    const all = curated.blocks.flatMap((b) => names(b.node))
    // Taken params are not rendered twice; everything else still is.
    for (const name of ["prompt", "negative_prompt", "styles", "script"]) expect(all).not.toContain(name)
    for (const name of ["width", "height", "batch_count", "cfg_scale", "enable_hr", "hr_prompt", "override_settings"]) expect(all).toContain(name)
    expect(names(curated.blocks.find((b) => b.kind === "sections")!.node)).toContain("denoising_strength")
  })

  it("follows the user's category order", () => {
    const layout = (txt2imgContract as unknown as TabSchema).core.layout
    if (layout.type !== "column") throw new Error("expected a column")
    const reversed = { ...layout, children: [...layout.children].reverse() }
    const kinds = curateCore(layout).blocks.map((b) => b.kind)
    expect(curateCore(reversed).blocks.map((b) => b.kind)).toEqual([...kinds].reverse())
  })

  it("puts the img2img source and inpaint options first", () => {
    const curated = curateCore((img2imgContract as unknown as TabSchema).core.layout)
    const sources = curated.blocks.filter((b) => b.kind === "source").flatMap((b) => names(b.node))
    expect(sources).toEqual(expect.arrayContaining(["mode", "init_img", "sketch", "mask", "resize_mode", "mask_blur", "inpainting_fill"]))
    expect(curated.blocks[0]?.kind).toBe("source")
    const guidance = curated.blocks.filter((b) => b.kind === "guidance").flatMap((b) => names(b.node))
    expect(guidance).toEqual(expect.arrayContaining(["cfg_scale", "denoising_strength"]))
  })

  it("keeps the mock form's nodes", () => {
    const curated = curateCore(tabSchema("txt2img").core.layout)
    expect(curated.prompt).not.toBeNull()
    expect(curated.blocks.some((b) => b.kind === "size")).toBe(true)
    expect(curated.blocks.flatMap((b) => names(b.node))).not.toContain("script")
  })
})
