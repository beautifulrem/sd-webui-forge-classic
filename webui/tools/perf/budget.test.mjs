import { describe, expect, it } from "vitest"
import { analyse, check, classify, closure, entryFiles, firstView } from "./budget.mjs"

// A small build: index.html loads main (which imports vendor); two routes share a form chunk; an
// editor and a widget are lazy.
const manifest = {
  "index.html": { file: "assets/main.js", isEntry: true, imports: ["_vendor.js"] },
  "_vendor.js": { file: "assets/vendor.js" },
  "_form.js": { file: "assets/form.js", imports: ["_vendor.js"] },
  "src/pages/generate/index.tsx": { file: "assets/generate.js", isDynamicEntry: true, imports: ["_form.js"] },
  "src/pages/settings/index.tsx": { file: "assets/settings.js", isDynamicEntry: true, imports: ["_form.js"] },
  "src/canvas/ui/Editor.tsx": { file: "assets/editor.js", isDynamicEntry: true },
  "src/widgets/big/index.tsx": { file: "assets/big.js", isDynamicEntry: true, imports: ["_form.js"] },
}
const html = '<script type="module" crossorigin src="/assets/main.js"></script>'
const KB = { "assets/main.js": 100, "assets/vendor.js": 150, "assets/form.js": 90, "assets/generate.js": 70, "assets/settings.js": 20, "assets/editor.js": 400, "assets/big.js": 65 }
const size = (file) => ({ raw: KB[file] * 3072, gzip: KB[file] * 1024, brotli: KB[file] * 900 })
const budget = {
  initialKB: 300,
  routeKB: 150,
  lazyKB: 150,
  defaultRoute: "src/pages/generate/index.tsx",
  exempt: { "src/canvas/ui/Editor.tsx": "editor" },
  exceptions: { "src/pages/generate/index.tsx": { maxKB: 170, withInitialMaxKB: 420, reason: "default route" } },
}

describe("bundle budget", () => {
  it("finds the initial JS from index.html and the static imports", () => {
    expect(entryFiles(html)).toEqual(["assets/main.js"])
    expect([...closure(manifest, ["assets/main.js"])].sort()).toEqual(["assets/main.js", "assets/vendor.js"])
  })

  it("reports what each lazy entry adds on top of the initial JS", () => {
    const report = analyse(manifest, html, size, budget)
    expect(report.initial.gzipKB).toBe(250)
    const added = Object.fromEntries(report.entries.map((entry) => [entry.key, [entry.kind, entry.addedKB]]))
    expect(added).toEqual({
      "src/canvas/ui/Editor.tsx": ["exempt", 400],
      "src/pages/generate/index.tsx": ["default-route", 160],
      "src/widgets/big/index.tsx": ["lazy", 155],
      "src/pages/settings/index.tsx": ["route", 110],
    })
    expect(report.defaultRoute).toEqual({ key: "src/pages/generate/index.tsx", addedKB: 160, withInitialKB: 410, parts: [], withPartsAddedKB: 160 })
    // a lazy part the route fetches as it renders counts with it (shared files once)
    const withParts = analyse(manifest, html, size, { ...budget, defaultRouteParts: ["src/widgets/big/index.tsx"] })
    expect(withParts.defaultRoute).toMatchObject({ addedKB: 160, parts: ["src/widgets/big/index.tsx"], withPartsAddedKB: 225 })
  })

  it("classifies routes, the default route, exempt editors and other lazy entries", () => {
    expect(classify("src/pages/queue/index.tsx", budget)).toBe("route")
    expect(classify("src/pages/settings/system/ServerView.tsx", budget)).toBe("lazy")
    expect(classify("src/pages/generate/index.tsx", budget)).toBe("default-route")
    expect(classify("src/canvas/ui/Editor.tsx", budget)).toBe("exempt")
  })

  it("flags entries over budget, honouring exemptions and capped exceptions", () => {
    const report = analyse(manifest, html, size, budget)
    // the widget is over the lazy budget; the exempt editor and the default route within its cap are not
    expect(check(report, budget)).toEqual(["src/widgets/big/index.tsx adds 155 KB > 150 KB"])
    expect(check(report, { ...budget, lazyKB: 160, exceptions: { "src/pages/generate/index.tsx": { maxKB: 150, withInitialMaxKB: 400 } } })).toEqual([
      "src/pages/generate/index.tsx adds 160 KB > 150 KB (its recorded exception)",
      "initial JS + src/pages/generate/index.tsx: 410 KB > 400 KB (its recorded exception)",
    ])
    expect(check(report, { ...budget, initialKB: 200, lazyKB: 160 })).toEqual(["initial JS 250 KB > 200 KB"])
  })

  it("rejects a budget naming entries the build does not have", () => {
    const report = analyse(manifest, html, size, budget)
    expect(check(report, { ...budget, lazyKB: 160, exempt: { "src/gone.tsx": "removed" } })).toEqual(["src/gone.tsx: named in the budget but not a lazily loaded entry of this build"])
    expect(check(report, { ...budget, lazyKB: 160, defaultRouteParts: ["src/gone.tsx"] })).toEqual(["src/gone.tsx: named in the budget but not a lazily loaded entry of this build"])
  })

  it("splits a measured first view into initial, default route, editors and the rest", () => {
    const view = firstView(manifest, html, size, budget, ["assets/main.js", "assets/vendor.js", "assets/generate.js", "assets/form.js", "assets/editor.js", "assets/big.js"])
    expect(view).toEqual({ scripts: 6, totalKB: 875, withoutExemptKB: 475, initialKB: 250, defaultRouteKB: 160, exemptKB: 400, otherKB: 65 })
    // the route's form part, fetched as it renders, is the route's
    const parts = firstView(manifest, html, size, { ...budget, defaultRouteParts: ["src/widgets/big/index.tsx"] }, ["assets/main.js", "assets/vendor.js", "assets/generate.js", "assets/form.js", "assets/editor.js", "assets/big.js"])
    expect(parts).toMatchObject({ defaultRouteKB: 225, otherKB: 0 })
  })
})
