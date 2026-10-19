/**
 * Every chrome string must have a zh-CN translation. Keys are found the way
 * docs/remi-webui/frontend.md asks them to be written: literal `t("…")`
 * calls, plus `label:` / `group:` / `title=` / `description=` literals in the
 * files that pass English strings to `t()` indirectly.
 */
import { readdirSync, readFileSync } from "node:fs"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import zh from "./zh-CN.json"

const sources = import.meta.glob<string>(["../**/*.{ts,tsx}", "!../**/*.test.*", "!../api/mock/**"], {
  query: "?raw",
  import: "default",
  eager: true,
})

/** Files whose English label constants are translated at render time. */
const INDIRECT = /(app\/(routes|App|AppearanceMenu|Header)\.tsx|pages\/.*\.tsx?|canvas\/.*\.tsx?|components\/PagePlaceholder\.tsx)$/

function chromeKeys(): Set<string> {
  const keys = new Set<string>()
  for (const [path, source] of Object.entries(sources)) {
    // `t("…", { ns: … })` names a widget's own catalog (checked by the widget's tests), not chrome;
    // `t("…", { context: "x" })` is the key "…_x" (one English word, two meanings in Chinese).
    for (const m of source.matchAll(/\bt\(\s*"((?:[^"\\]|\\.)+)"(\s*,\s*\{\s*(?:ns:|context:\s*"(\w+)"))?/g)) {
      if (m[3]) keys.add(`${m[1]}_${m[3]}`)
      else if (!m[2]) keys.add(m[1]!)
    }
    if (INDIRECT.test(path)) {
      for (const m of source.matchAll(/\b(?:label|title|description): "([^"]+)"/g)) keys.add(m[1]!)
      for (const m of source.matchAll(/\b(?:group): "([A-Z][^"]+)"/g)) keys.add(m[1]!)
      for (const m of source.matchAll(/\b(?:title|description)="([^"]+)"/g)) keys.add(m[1]!)
    }
  }
  return keys
}

describe("zh-CN chrome catalog", () => {
  it("translates every chrome string", () => {
    const catalog = zh as Record<string, string>
    const keys = chromeKeys()
    expect(keys.size).toBeGreaterThan(100) // the scan itself works
    const missing = [...keys].filter((key) => !catalog[key])
    expect(missing).toEqual([])
  })

  it("keeps interpolation placeholders intact", () => {
    for (const [key, value] of Object.entries(zh as Record<string, string>)) {
      const vars = (s: string) => [...s.matchAll(/\{\{(\w+)\}\}/g)].map((m) => m[1]).sort()
      expect(vars(value), key).toEqual(vars(key))
    }
  })
})

/** The server's zh_CN catalog as `/api/v2/i18n/zh_CN` merges it: core, then every built-in extension's. */
function serverCatalog(): Record<string, string> {
  const root = join(process.cwd(), "..")
  const files = [join(root, "localizations", "zh_CN.json")]
  for (const extension of readdirSync(join(root, "extensions-builtin")).sort()) {
    const path = join(root, "extensions-builtin", extension, "localizations", "zh_CN.json")
    try {
      readFileSync(path)
      files.push(path)
    } catch {
      // no catalog
    }
  }
  const merged: Record<string, string> = {}
  for (const path of files) Object.assign(merged, JSON.parse(readFileSync(path, "utf8")))
  return merged
}

describe("zh-CN chrome vs server catalogs", () => {
  it("never overrides a server translation with a different one", () => {
    // t() looks in the chrome catalog first: a chrome key that the server translates
    // differently changes the server's labels too (e.g. a theme's "System" renaming the
    // "System" settings category). Same meaning → same wording; another meaning → a context key.
    const server = serverCatalog()
    expect(Object.keys(server).length).toBeGreaterThan(1000) // the catalogs were found
    const clashes = Object.entries(zh as Record<string, string>)
      .filter(([key, value]) => key in server && server[key] !== value)
      .map(([key, value]) => `${key}: chrome "${value}" vs server "${server[key]}"`)
    expect(clashes).toEqual([])
  })
})

