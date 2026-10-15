/**
 * Every chrome string must have a zh-CN translation. Keys are found the way
 * docs/remi-webui/frontend.md asks them to be written: literal `t("…")`
 * calls, plus `label:` / `group:` / `title=` / `description=` literals in the
 * files that pass English strings to `t()` indirectly.
 */
import { describe, expect, it } from "vitest"
import zh from "./zh-CN.json"

const sources = import.meta.glob<string>(["../**/*.{ts,tsx}", "!../**/*.test.*", "!../api/mock/**", "!../components/ui/**"], {
  query: "?raw",
  import: "default",
  eager: true,
})

/** Files whose English label constants are translated at render time. */
const INDIRECT = /(app\/(routes|App|AppearanceMenu|Header)|pages\/.*|components\/PagePlaceholder)\.tsx$/

function chromeKeys(): Set<string> {
  const keys = new Set<string>()
  for (const [path, source] of Object.entries(sources)) {
    for (const m of source.matchAll(/\bt\(\s*"((?:[^"\\]|\\.)+)"/g)) keys.add(m[1]!)
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
