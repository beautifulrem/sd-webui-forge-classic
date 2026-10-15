// Bundle budget report (docs/research/ui-performance.md).
//
//   pnpm exec vite build --manifest --outDir /tmp/remi-perf-dist --emptyOutDir
//   node tools/perf/bundle.mjs /tmp/remi-perf-dist [--json out.json]
//
// Initial JS = the scripts index.html loads plus their static imports. Each lazy entry
// (route page, widget, editor) is reported with its own chunk and with what it adds on top
// of the initial set (its chunk plus static imports not loaded yet). Sizes are gzip -6 and
// brotli -q11 of the built files.
import { readFileSync, writeFileSync } from "node:fs"
import path from "node:path"
import { brotliCompressSync, constants, gzipSync } from "node:zlib"

const [dist = "dist", ...rest] = process.argv.slice(2)
const jsonOut = rest[0] === "--json" ? rest[1] : undefined
const manifest = JSON.parse(readFileSync(path.join(dist, ".vite/manifest.json"), "utf8"))
const byFile = new Map(Object.values(manifest).map((entry) => [entry.file, entry]))
const keyToFile = new Map(Object.entries(manifest).map(([key, entry]) => [key, entry.file]))

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

function closure(files) {
  const seen = new Set()
  const stack = [...files]
  while (stack.length) {
    const file = stack.pop()
    if (seen.has(file)) continue
    seen.add(file)
    for (const key of byFile.get(file)?.imports ?? []) stack.push(keyToFile.get(key))
  }
  return seen
}

const sum = (files, kind = "gzip") => [...files].filter((f) => f.endsWith(".js")).reduce((n, f) => n + size(f)[kind], 0)
const kb = (bytes) => (bytes / 1024).toFixed(1)

const html = readFileSync(path.join(dist, "index.html"), "utf8")
const entryFiles = [...html.matchAll(/<script[^>]+src="\/?([^"]+\.js)"/g)].map((m) => m[1])
const cssFiles = [...html.matchAll(/<link[^>]+rel="stylesheet"[^>]+href="\/?([^"]+\.css)"/g)].map((m) => m[1])
const initial = closure(entryFiles)

const lazy = Object.entries(manifest)
  .filter(([, entry]) => entry.isDynamicEntry && entry.file.endsWith(".js"))
  .map(([key, entry]) => {
    const extra = [...closure([entry.file])].filter((file) => !initial.has(file))
    return { key, file: entry.file, chunk: size(entry.file), added: { gzip: sum(extra), brotli: sum(extra, "brotli"), files: extra.length } }
  })
  .sort((a, b) => b.added.gzip - a.added.gzip)

const chunks = [...byFile.keys()].filter((f) => f.endsWith(".js")).map((file) => ({ file, ...size(file) }))
chunks.sort((a, b) => b.gzip - a.gzip)

const report = {
  initial: { files: [...initial], gzip: sum(initial), brotli: sum(initial, "brotli"), raw: sum(initial, "raw"), css: cssFiles.map((f) => ({ file: f, ...size(f) })) },
  lazy,
  largest: chunks.slice(0, 15),
  totalJs: { files: chunks.length, gzip: chunks.reduce((n, c) => n + c.gzip, 0) },
}

console.log(`Initial JS: ${kb(report.initial.gzip)} KB gzip, ${kb(report.initial.brotli)} KB brotli, ${kb(report.initial.raw)} KB raw (${initial.size} files)`)
for (const file of initial) console.log(`  ${file.padEnd(48)} ${kb(size(file).gzip).padStart(7)} KB gzip`)
for (const css of report.initial.css) console.log(`  ${css.file.padEnd(48)} ${kb(css.gzip).padStart(7)} KB gzip (CSS)`)
console.log(`\nLazy entries (own chunk / added on top of the initial JS, gzip KB):`)
for (const item of lazy.slice(0, 25)) console.log(`  ${item.key.replace(/^src\//, "").padEnd(52)} ${kb(item.chunk.gzip).padStart(7)} ${kb(item.added.gzip).padStart(8)}  (${item.added.files} files)`)
console.log(`\nLargest chunks (gzip KB):`)
for (const chunk of report.largest) console.log(`  ${chunk.file.padEnd(48)} ${kb(chunk.gzip).padStart(7)}`)
console.log(`\nAll JS: ${report.totalJs.files} files, ${kb(report.totalJs.gzip)} KB gzip`)
if (jsonOut) writeFileSync(jsonOut, JSON.stringify(report, null, 1))
