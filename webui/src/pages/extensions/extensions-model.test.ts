import { describe, expect, it, vi } from "vitest"
import {
  dirnameFromUrl,
  disabledList,
  type ExtensionInfo,
  filterIndex,
  formatDate,
  type IndexEntry,
  type IndexFilter,
  isGitUrl,
  isValidDirname,
  NO_CHANGES,
  pruneStaged,
  stageEnabled,
  stagedCount,
  stageUpdate,
} from "./extensions-model"
import { runChecks } from "./update-check"

const entry = (name: string, extra: Partial<IndexEntry> = {}): IndexEntry => ({ name, url: `https://github.com/x/${name}`, description: "", tags: [], stars: 0, order: 0, installed: false, ...extra })

const items: IndexEntry[] = [
  entry("Tagger", { tags: ["tab", "ads"], added: "2023-01-02", stars: 50, order: 0, description: "WD14 tagging", commit_time: "2024-05-01T00:00:00Z" }),
  entry("Browser", { tags: ["tab"], added: "2022-06-01", stars: 900, order: 1, installed: true, commit_time: "2025-01-01T00:00:00Z" }),
  entry("Zh pack", { tags: ["localization"], added: "2024-03-03", stars: 5, order: 2 }),
  entry("Pose", { tags: ["script", "editing"], added: "2021-12-12", stars: 200, order: 3 }),
]

const base: IndexFilter = { query: "", tags: new Set(), tagMode: "hide", match: "any", sort: "index" }
const names = (list: IndexEntry[]) => list.map((e) => e.name)

describe("extension index", () => {
  it("searches name, description, URL and tags", () => {
    expect(names(filterIndex(items, { ...base, query: "wd14" }))).toEqual(["Tagger"])
    expect(names(filterIndex(items, { ...base, query: "x/pose" }))).toEqual(["Pose"])
    expect(names(filterIndex(items, { ...base, query: "editing script" }))).toEqual(["Pose"])
  })

  it("hides or shows tagged entries, with installed as a tag", () => {
    const hidden = new Set(["ads", "localization", "installed"])
    expect(names(filterIndex(items, { ...base, tags: hidden }))).toEqual(["Pose"])
    expect(names(filterIndex(items, { ...base, tags: new Set(["installed"]), tagMode: "show" }))).toEqual(["Browser"])
    expect(names(filterIndex(items, { ...base, tags: new Set(["tab", "ads"]), tagMode: "show", match: "all" }))).toEqual(["Tagger"])
    expect(names(filterIndex(items, { ...base, tags: new Set(["tab", "ads"]), tagMode: "show", match: "any" }))).toEqual(["Tagger", "Browser"])
  })

  it("sorts like the former tab", () => {
    expect(names(filterIndex(items, { ...base, sort: "newest" }))).toEqual(["Zh pack", "Tagger", "Browser", "Pose"])
    expect(names(filterIndex(items, { ...base, sort: "oldest" }))).toEqual(["Pose", "Browser", "Tagger", "Zh pack"])
    expect(names(filterIndex(items, { ...base, sort: "a-z" }))).toEqual(["Browser", "Pose", "Tagger", "Zh pack"])
    expect(names(filterIndex(items, { ...base, sort: "z-a" }))).toEqual(["Zh pack", "Tagger", "Pose", "Browser"])
    expect(names(filterIndex(items, { ...base, sort: "stars" }))).toEqual(["Browser", "Pose", "Tagger", "Zh pack"])
    expect(names(filterIndex(items, { ...base, sort: "updated" }))).toEqual(["Browser", "Tagger", "Zh pack", "Pose"])
  })

  it("does not reorder the input array", () => {
    const copy = [...items]
    filterIndex(items, { ...base, sort: "a-z" })
    expect(items).toEqual(copy)
  })

  it("formats dates and keeps unreadable ones", () => {
    expect(formatDate(null)).toBe("")
    expect(formatDate("soon")).toBe("soon")
    expect(formatDate("2024-03-03T00:00:00Z", "en")).toMatch(/2024/)
  })
})

describe("install URLs", () => {
  it("accepts git URLs only and derives the folder", () => {
    expect(isGitUrl("https://github.com/a/b")).toBe(true)
    expect(isGitUrl("git@github.com:a/b.git")).toBe(true)
    expect(isGitUrl("file:///etc")).toBe(false)
    expect(isGitUrl("github.com/a/b")).toBe(false)
    expect(dirnameFromUrl("https://github.com/a/sd-ext.git/")).toBe("sd-ext")
    expect(dirnameFromUrl("git@github.com:a/sd-ext.git")).toBe("sd-ext")
    expect(isValidDirname("sd-ext_2.1")).toBe(true)
    expect(isValidDirname("../up")).toBe(false)
  })
})

describe("staged changes", () => {
  const ext = (name: string, extra: Partial<ExtensionInfo> = {}): ExtensionInfo => ({ name, enabled: true, builtin: false, can_update: false, ...extra })
  const list = [ext("a"), ext("b", { enabled: false }), ext("c", { can_update: true })]

  it("tracks switches that differ from the server and updates to apply", () => {
    let staged = stageEnabled(NO_CHANGES, list[0]!, false)
    staged = stageEnabled(staged, list[1]!, true)
    staged = stageUpdate(staged, "c", true)
    expect(stagedCount(staged)).toBe(3)
    expect(disabledList(list, staged)).toEqual(["a"])
    // switching back is no change
    staged = stageEnabled(staged, list[0]!, true)
    expect(staged.enabled).toEqual({ b: true })
    expect(disabledList(list, NO_CHANGES)).toBeNull()
  })

  it("prunes changes that no longer apply", () => {
    const staged = { enabled: { a: false, gone: true, b: false }, update: ["c", "a"] }
    expect(pruneStaged(staged, list)).toEqual({ enabled: { a: false }, update: ["c"] })
  })
})

describe("update checks", () => {
  it("runs a few at a time, reports progress and collects failures", async () => {
    let running = 0
    let peak = 0
    const progress: number[] = []
    const failed = await runChecks(
      ["a", "b", "c", "d", "e"],
      async (name) => {
        running++
        peak = Math.max(peak, running)
        await new Promise((resolve) => setTimeout(resolve, 5))
        running--
        if (name === "c") throw new Error("offline")
      },
      { concurrency: 2, onProgress: (p) => progress.push(p.done) },
    )
    expect(failed).toEqual(["c"])
    expect(peak).toBe(2)
    expect(progress.at(-1)).toBe(5)
  })

  it("stops scheduling when aborted", async () => {
    const controller = new AbortController()
    const check = vi.fn(async (name: string) => {
      if (name === "b") controller.abort()
    })
    await runChecks(["a", "b", "c", "d"], check, { concurrency: 1, signal: controller.signal })
    expect(check.mock.calls.map(([name]) => name)).toEqual(["a", "b"])
  })
})

describe("server error messages", () => {
  const t = (key: string, options?: Record<string, unknown>) =>
    ({ "git {{command}} failed: {{reason}}": "git {{command}} 失败：{{reason}}" })[key]?.replace(/\{\{(\w+)\}\}/g, (_m, name: string) => String(options?.[name])) ??
    String(options?.defaultValue ?? key)

  it("translates the template with its params, and falls back to the English detail", async () => {
    const { ApiError } = await import("@/api/http")
    const { adminErrorMessage } = await import("./extensions-model")
    const failed = new ApiError(502, "git fetch failed: offline", { detail: "git fetch failed: offline", template: "git {{command}} failed: {{reason}}", params: { command: "fetch", reason: "offline" } })
    expect(adminErrorMessage(failed, t)).toBe("git fetch 失败：offline")
    const unknown = new ApiError(422, "the index is too large", { detail: "the index is too large", template: "the index is too large", params: {} })
    expect(adminErrorMessage(unknown, t)).toBe("the index is too large")
    expect(adminErrorMessage(new Error("plain"), t)).toBe("plain")
  })
})
