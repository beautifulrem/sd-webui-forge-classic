import { afterEach, describe, expect, it, vi } from "vitest"
import { promptToolsApi } from "./api"
import { promptFromTranslation, translatePromptToEnglish } from "./translate"

describe("writing the prompt in English", () => {
  afterEach(() => vi.restoreAllMocks())

  it("replaces every foreign tag on one parse, even when a translation holds commas", async () => {
    const translate = vi.spyOn(promptToolsApi, "translate").mockResolvedValue({ provider: "openai", translations: ["girl, smiling.", "cat (animal)", null, "dog"] })
    const result = await translatePromptToEnglish("微笑的女孩, (猫:1.2), 未知, smile, /*狗, */<lora:x:1>")
    expect(translate).toHaveBeenCalledWith(["微笑的女孩", "猫", "未知", "狗"], "en")
    expect(result).toEqual({ text: "girl, smiling, (cat \\(animal\\):1.2), 未知, smile, /*dog, */<lora:x:1>", translated: 3 })
  })

  it("leaves an English prompt alone", async () => {
    const translate = vi.spyOn(promptToolsApi, "translate")
    expect(await translatePromptToEnglish("1girl, smile")).toEqual({ text: "1girl, smile", translated: 0 })
    expect(translate).not.toHaveBeenCalled()
  })

  it("writes a translation as prompt text", () => {
    expect(promptFromTranslation(" Girl、smiling；happy\nAND sad。")).toBe("Girl, smiling, happy, and sad")
    expect(promptFromTranslation("a BREAK b")).toBe("a break b")
    expect(promptFromTranslation("x [y]")).toBe("x \\[y\\]")
  })
})

describe("API chunks", () => {
  afterEach(() => vi.restoreAllMocks())

  it("splits long lists into requests the server accepts", async () => {
    const { client } = await import("@/api/http")
    const { chunks } = await import("./api")
    expect(chunks([1, 2, 3, 4, 5], 2)).toEqual([[1, 2], [3, 4], [5]])
    const post = vi.spyOn(client, "POST").mockImplementation((async (path: string, init: { body: { tags?: string[]; texts?: string[] } }) => {
      const data = path.endsWith("/lookup")
        ? { tags: Object.fromEntries(init.body.tags!.map((tag) => [tag, { category: 0, count: 1, translation: null }])) }
        : { provider: "dictionary", translations: init.body.texts!.map((text) => text.toUpperCase()) }
      return { data, response: new Response(null, { status: 200 }) }
    }) as never)
    const tags = Array.from({ length: 1201 }, (_, i) => `tag${i}`)
    expect(Object.keys(await promptToolsApi.lookup(tags))).toHaveLength(1201)
    expect(post.mock.calls.map((call) => (call[1] as { body: { tags: string[] } }).body.tags.length)).toEqual([500, 500, 201])
    post.mockClear()
    const answer = await promptToolsApi.translate(tags.slice(0, 450), "en")
    expect(answer.translations).toHaveLength(450)
    expect(answer.translations[449]).toBe("TAG449")
    expect(post.mock.calls.map((call) => (call[1] as { body: { texts: string[] } }).body.texts.length)).toEqual([200, 200, 50])
  })
})
