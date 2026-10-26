// Bundle budget report and check (docs/research/ui-performance.md, "Bundle").
//
//   pnpm exec vite build --manifest --outDir /tmp/remi-perf-dist --emptyOutDir
//   node tools/perf/bundle.mjs /tmp/remi-perf-dist [--json out.json] [--check [budget.json]] [--load browser-load.json]
//
// Initial JS = the scripts index.html loads plus their static imports. Each lazily loaded entry
// (route page, widget, dialog, editor) is reported with its own chunk and with what it adds on top
// of the initial set (its chunk plus static imports not loaded yet): what a first visit to that
// route downloads. Sizes are gzip -6 and brotli -q11 of the built files.
//
// --check applies tools/perf/budget.json (or the given file): the initial JS, every route and
// every other lazy entry against their limits, the exempt editor chunks and the recorded
// exceptions (each with its own cap); exits 1 on a violation (tools/check.sh, step perf:bundle).
// --load splits the scripts a measured page load fetched (browser.mjs --only load) into the
// initial JS, the default route, the exempt editors and the rest of the first view.
import { existsSync, readFileSync, writeFileSync } from "node:fs"
import path from "node:path"
import { brotliCompressSync, constants, gzipSync } from "node:zlib"
import { analyse, check, firstView } from "./budget.mjs"

const argv = process.argv.slice(2)
const dist = argv[0] && !argv[0].startsWith("--") ? argv[0] : "dist"
const option = (name) => {
  const index = argv.indexOf(`--${name}`)
  if (index < 0) return undefined
  const value = argv[index + 1]
  return value && !value.startsWith("--") ? value : true
}
const jsonOut = option("json")
const checkOption = option("check")
const loadFile = option("load")
const budgetFile = typeof checkOption === "string" ? checkOption : path.join(import.meta.dirname, "budget.json")
const budget = JSON.parse(readFileSync(budgetFile, "utf8"))

const manifest = JSON.parse(readFileSync(path.join(dist, ".vite/manifest.json"), "utf8"))
const html = readFileSync(path.join(dist, "index.html"), "utf8")

const sizes = new Map()
function size(file) {
  if (!sizes.has(file)) {
    const raw = readFileSync(path.join(dist, file))
    sizes.set(file, {
      raw: raw.length,
      gzip: gzipSync(raw, { level: 6 }).length,
      brotli: brotliCompressSync(raw, { params: { [constants.BROTLI_PARAM_QUALITY]: 11 } }).length,
    })
  }
  return sizes.get(file)
}
const kb = (bytes) => (bytes / 1024).toFixed(1)

const report = analyse(manifest, html, size, budget)
const cssFiles = [...html.matchAll(/<link[^>]+rel="stylesheet"[^>]+href="\/?([^"]+\.css)"/g)].map((m) => m[1])
const chunks = Object.values(manifest)
  .map((entry) => entry.file)
  .filter((file, index, all) => file.endsWith(".js") && all.indexOf(file) === index)
  .map((file) => ({ file, ...size(file) }))
  .sort((a, b) => b.gzip - a.gzip)
report.initial.rawKB = Number(kb(report.initial.files.reduce((n, f) => n + size(f).raw, 0)))
report.initial.css = cssFiles.map((file) => ({ file, ...size(file) }))
report.largest = chunks.slice(0, 15)
report.totalJs = { files: chunks.length, gzip: chunks.reduce((n, c) => n + c.gzip, 0) }

console.log(`Initial JS: ${report.initial.gzipKB} KB gzip, ${report.initial.brotliKB} KB brotli, ${report.initial.rawKB} KB raw (${report.initial.files.length} files; budget ${budget.initialKB} KB)`)
for (const file of report.initial.files) console.log(`  ${file.padEnd(48)} ${kb(size(file).gzip).padStart(7)} KB gzip`)
for (const css of report.initial.css) console.log(`  ${css.file.padEnd(48)} ${kb(css.gzip).padStart(7)} KB gzip (CSS)`)
if (report.defaultRoute) {
  const { key, addedKB, withInitialKB, parts, withPartsAddedKB } = report.defaultRoute
  console.log(`Default route ${key}: adds ${addedKB} KB (with the initial JS ${withInitialKB} KB)${parts.length ? `; with the lazy parts its first render fetches (${parts.join(", ")}) ${withPartsAddedKB} KB` : ""}`)
}
console.log(`\nLazy entries (own chunk / added on top of the initial JS, gzip KB; kind):`)
for (const entry of report.entries.slice(0, 30)) console.log(`  ${entry.key.replace(/^src\//, "").slice(0, 60).padEnd(60)} ${String(entry.chunkKB).padStart(6)} ${String(entry.addedKB).padStart(7)}  ${entry.kind} (${entry.addedFiles} files)`)
console.log(`\nLargest chunks (gzip KB):`)
for (const chunk of report.largest) console.log(`  ${chunk.file.padEnd(48)} ${kb(chunk.gzip).padStart(7)}`)
console.log(`\nAll JS: ${report.totalJs.files} files, ${kb(report.totalJs.gzip)} KB gzip`)

if (loadFile) {
  const load = JSON.parse(readFileSync(loadFile, "utf8")).load
  report.firstView = {}
  for (const [tab, run] of Object.entries(load)) {
    const files = run.scripts.files.map((script) => script.file.replace(/^\//, ""))
    const missing = files.filter((file) => !existsSync(path.join(dist, file)))
    if (missing.length) throw new Error(`${loadFile}: ${missing.length} scripts are not in ${dist} (not the build that was served), e.g. ${missing[0]}`)
    report.firstView[tab] = firstView(manifest, html, size, budget, files)
    const v = report.firstView[tab]
    console.log(`\nFirst view ${tab} (measured, ${v.scripts} scripts): ${v.totalKB} KB = initial ${v.initialKB} + default route ${v.defaultRouteKB} + exempt editors ${v.exemptKB} + other ${v.otherKB}; without the editors ${v.withoutExemptKB} KB`)
  }
}

if (jsonOut) writeFileSync(jsonOut, JSON.stringify(report, null, 1))

if (checkOption) {
  const problems = check(report, budget)
  const exceptions = Object.entries(budget.exceptions ?? {})
  console.log(`\nBudget (${path.relative(process.cwd(), budgetFile)}): initial ${budget.initialKB} KB, routes ${budget.routeKB} KB, other lazy entries ${budget.lazyKB} KB; exempt: ${Object.keys(budget.exempt ?? {}).length} editor entries; exceptions: ${exceptions.map(([key, value]) => `${key} (${value.maxKB} KB)`).join(", ") || "none"}`)
  if (problems.length) {
    for (const problem of problems) console.error(`  OVER BUDGET: ${problem}`)
    process.exit(1)
  }
  console.log("  every entry is within its budget")
}
