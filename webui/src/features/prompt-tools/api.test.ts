/** The `/api/v2/prompt-tools` client: request shapes, defaults, chunking and repeated query keys. */
import { afterEach, describe, expect, it, vi } from "vitest"
import { declaredInOpenApi, recordFetch } from "@/test/fetch-recorder"
import { chunks, LOOKUP_CHUNK, promptToolsApi, TRANSLATE_CHUNK } from "./api"


describe("prompt-tools api", () => {
  afterEach(() => vi.restoreAllMocks())

  it("splits lists into chunks of at most the size", () => {
    expect(chunks([1, 2, 3, 4, 5], 2)).toEqual([[1, 2], [3, 4], [5]])
    expect(chunks([], 3)).toEqual([])
    expect(chunks([1, 2], 2)).toEqual([[1, 2]])
  })

  it("looks up any number of tags in server-sized chunks and merges the answers", async () => {
    const tags = Array.from({ length: LOOKUP_CHUNK + 3 }, (_, i) => `tag${i}`)
    const fetch = recordFetch(({ body }) => {
      const part = (body as { tags: string[] }).tags
      return { body: { tags: Object.fromEntries(part.map((tag) => [tag, { name: tag, category: 0, post_count: 1 }])) } }
    })
    const result = await promptToolsApi.lookup(tags)
    expect(fetch.requests.map((r) => (r.body as { tags: string[] }).tags.length)).toEqual([LOOKUP_CHUNK, 3])
    expect(fetch.requests.every((r) => r.method === "POST" && r.path === "/api/v2/prompt-tools/lookup")).toBe(true)
    expect(Object.keys(result)).toHaveLength(tags.length)
    expect(result.tag502).toMatchObject({ name: "tag502" })
  })

  it("translates in chunks and keeps the order of the texts", async () => {
    const texts = Array.from({ length: TRANSLATE_CHUNK + 1 }, (_, i) => `t${i}`)
    const fetch = recordFetch(({ body }) => {
      const { texts: part, target } = body as { texts: string[]; target: string }
      return { body: { provider: "deepl", translations: part.map((text) => (text === "t1" ? null : `${target}:${text}`)) } }
    })
    const result = await promptToolsApi.translate(texts, "zh")
    expect(fetch.requests).toHaveLength(2)
    expect(result.provider).toBe("deepl")
    expect(result.translations).toHaveLength(texts.length)
    expect(result.translations.slice(0, 3)).toEqual(["zh:t0", null, "zh:t2"])
    expect(result.translations.at(-1)).toBe(`zh:t${TRANSLATE_CHUNK}`)
    // nothing to translate: no request, no provider
    expect(await promptToolsApi.translate([], "zh")).toEqual({ provider: "", translations: [] })
    expect(fetch.requests).toHaveLength(2)
  })

  it("completes with negative off by default and returns the items", async () => {
    const fetch = recordFetch(() => ({ body: { items: [{ name: "cat", type: "tag" }] } }))
    expect(await promptToolsApi.complete({ mode: "tag", q: "ca", limit: 5 })).toEqual([{ name: "cat", type: "tag" }])
    expect(Object.fromEntries(fetch.last().query)).toEqual({ mode: "tag", q: "ca", limit: "5", negative: "false" })
    await promptToolsApi.complete({ mode: "wildcard-value", q: "", negative: true, wildcard: "hair" })
    expect(Object.fromEntries(fetch.last().query)).toEqual({ mode: "wildcard-value", negative: "true", wildcard: "hair" })
  })

  it("lists favorites with a repeated kind key and without unset values", async () => {
    const fetch = recordFetch(() => ({ body: { items: [], next_cursor: null } }))
    await promptToolsApi.favorites.list({ kind: ["tags", "prompt"], collection: "", q: "x", cursor: null, limit: 20 })
    expect(fetch.last().query.getAll("kind")).toEqual(["tags", "prompt"])
    expect(fetch.last().query.toString()).toBe("kind=tags&kind=prompt&q=x&limit=20")
  })

  it("sends every other request as the server declares it, with the client's defaults", async () => {
    const fetch = recordFetch(({ path }) => {
      if (path.endsWith("/generate")) return { body: { prompt: "a cat, sitting" } }
      if (path.endsWith("/groups")) return { body: { categories: [{ name: "Pose" }] } }
      return undefined
    })
    const calls: [() => Promise<unknown>, string, string, unknown?][] = [
      [() => promptToolsApi.config(), "GET", "/api/v2/prompt-tools/config"],
      [() => promptToolsApi.recordUse({ name: "cat", type: "tag", negative: false }), "POST", "/api/v2/prompt-tools/usage", { name: "cat", type: "tag", negative: false }],
      [() => promptToolsApi.generate({ description: "a cat" }), "POST", "/api/v2/prompt-tools/generate", { description: "a cat", prompt: "", negative: false }],
      [() => promptToolsApi.setTranslationKey(""), "PUT", "/api/v2/prompt-tools/translation-key", { key: "" }],
      [() => promptToolsApi.groups("zh_CN"), "GET", "/api/v2/prompt-tools/groups"],
      [() => promptToolsApi.history.list({ q: "cat", field: "prompt" }), "GET", "/api/v2/prompt-tools/history"],
      [() => promptToolsApi.history.add({ prompt: "p", tab: "txt2img" }), "POST", "/api/v2/prompt-tools/history", { prompt: "p", negative_prompt: "", tab: "txt2img" }],
      [() => promptToolsApi.history.rename("h 1", "mine"), "PATCH", "/api/v2/prompt-tools/history/h%201", { name: "mine", field: null }],
      [() => promptToolsApi.history.rename("h1", "mine", "negative"), "PATCH", "/api/v2/prompt-tools/history/h1", { name: "mine", field: "negative" }],
      [() => promptToolsApi.history.remove("h1", "prompt"), "DELETE", "/api/v2/prompt-tools/history/h1"],
      [() => promptToolsApi.history.clear(), "DELETE", "/api/v2/prompt-tools/history"],
      [() => promptToolsApi.collections.list(), "GET", "/api/v2/prompt-tools/collections"],
      [() => promptToolsApi.collections.create("Faces"), "POST", "/api/v2/prompt-tools/collections", { name: "Faces" }],
      [() => promptToolsApi.collections.rename("c1", "Hands"), "PATCH", "/api/v2/prompt-tools/collections/c1", { name: "Hands" }],
      [() => promptToolsApi.collections.remove("c1"), "DELETE", "/api/v2/prompt-tools/collections/c1"],
      [() => promptToolsApi.favorites.add({ text: "cat", kind: "tags" }), "POST", "/api/v2/prompt-tools/favorites", { text: "cat", kind: "tags", name: "", collection_id: null }],
      [() => promptToolsApi.favorites.update("f1", { name: "Cat" }), "PATCH", "/api/v2/prompt-tools/favorites/f1", { name: "Cat" }],
      [() => promptToolsApi.favorites.remove("f1"), "DELETE", "/api/v2/prompt-tools/favorites/f1"],
      [() => promptToolsApi.favorites.import({ version: 1, collections: [], favorites: [] } as never), "POST", "/api/v2/prompt-tools/favorites-import", { version: 1, collections: [], favorites: [] }],
    ]
    for (const [call, method, path, body] of calls) {
      await call()
      const sent = fetch.last()
      expect({ method: sent.method, path: sent.path, body: sent.body }).toEqual({ method, path, body })
      expect(declaredInOpenApi(method, path), `${method} ${path} is not in openapi.json`).toBe(true)
    }
    expect(fetch.requests).toHaveLength(calls.length)
    // the answers are unwrapped where the client returns a field
    expect(await promptToolsApi.generate({ description: "x", prompt: "a cat" })).toBe("a cat, sitting")
    expect(await promptToolsApi.groups("en")).toEqual([{ name: "Pose" }])
    // query parameters: grouped history removal and the "keep named" clear
    await promptToolsApi.history.remove("h1", "prompt")
    expect(fetch.last().query.get("field")).toBe("prompt")
    await promptToolsApi.history.clear()
    expect(fetch.last().query.get("keep_named")).toBe("true")
    expect(promptToolsApi.favorites.exportUrl()).toBe("/api/v2/prompt-tools/favorites-export")
  })
})
