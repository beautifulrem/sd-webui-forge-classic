// Bundle budget analysis (bundle.mjs; tests in budget.test.mjs). Pure functions over a Vite
// manifest: no file system access (sizes come from `size(file)`).

/** The files `index.html` loads: its module scripts and their static imports. */
export function entryFiles(html) {
  return [...html.matchAll(/<script[^>]+src="\/?([^"]+\.js)"/g)].map((m) => m[1])
}

/** Static import closure of `files` in the manifest. */
export function closure(manifest, files) {
  const byFile = new Map(Object.values(manifest).map((entry) => [entry.file, entry]))
  const seen = new Set()
  const stack = [...files]
  while (stack.length) {
    const file = stack.pop()
    if (seen.has(file)) continue
    seen.add(file)
    for (const key of byFile.get(file)?.imports ?? []) stack.push(manifest[key].file)
  }
  return seen
}

const isJs = (file) => file.endsWith(".js")
const kbOf = (files, size, kind = "gzip") => [...files].filter(isJs).reduce((n, f) => n + size(f)[kind], 0) / 1024
const round1 = (value) => Math.round(value * 10) / 10

/** Which budget a lazily loaded entry falls under. */
export function classify(key, budget) {
  if (budget.exempt?.[key]) return "exempt"
  if (key === budget.defaultRoute) return "default-route"
  if (new RegExp(budget.routePattern ?? "^src/pages/[^/]+/index\\.tsx$").test(key)) return "route"
  return "lazy"
}

/**
 * The budget report: the initial JS (index.html's scripts and their static imports) and, per
 * lazily loaded entry, what it adds on top of the initial JS (its chunk and the static imports not
 * loaded yet) - what a first visit to that route (or the first use of that feature) downloads.
 */
export function analyse(manifest, html, size, budget) {
  const initial = closure(manifest, entryFiles(html))
  const entries = Object.entries(manifest)
    .filter(([, entry]) => entry.isDynamicEntry && isJs(entry.file))
    .map(([key, entry]) => {
      const added = [...closure(manifest, [entry.file])].filter((file) => !initial.has(file) && isJs(file))
      return { key, kind: classify(key, budget), file: entry.file, chunkKB: round1(size(entry.file).gzip / 1024), addedKB: round1(kbOf(added, size)), addedFiles: added.length }
    })
    .sort((a, b) => b.addedKB - a.addedKB)
  const route = entries.find((entry) => entry.key === budget.defaultRoute)
  const defaultRouteFiles = route ? closure(manifest, [route.file]) : new Set()
  // the route with the lazy parts its first render fetches (`defaultRouteParts`, e.g. its form)
  const viewFiles = route ? defaultView(manifest, budget) : new Set()
  return {
    initial: { files: [...initial].filter(isJs), gzipKB: round1(kbOf(initial, size)), brotliKB: round1(kbOf(initial, size, "brotli")) },
    defaultRoute: route
      ? {
          key: route.key,
          addedKB: route.addedKB,
          withInitialKB: round1(kbOf(new Set([...initial, ...defaultRouteFiles]), size)),
          parts: budget.defaultRouteParts ?? [],
          withPartsAddedKB: round1(kbOf([...viewFiles].filter((file) => !initial.has(file)), size)),
        }
      : null,
    entries,
  }
}

/** The default route's static closure and those of the lazy parts it fetches as it renders (`defaultRouteParts`). */
function defaultView(manifest, budget) {
  const keys = [budget.defaultRoute, ...(budget.defaultRouteParts ?? [])].filter((key) => manifest[key])
  return closure(
    manifest,
    keys.map((key) => manifest[key].file),
  )
}

/**
 * Budget violations of a report: the initial JS over `initialKB`; a route or another lazy entry
 * adding more than `routeKB` / `lazyKB`, unless it is exempt (the editors) or a recorded exception
 * within its own cap; the default route together with the lazy parts its first render fetches
 * (`defaultRouteParts`) adding more than `routeKB` (a split of the route into lazy parts that all
 * load at once must not hide its size); an exception or exemption naming an entry the build no
 * longer has.
 */
export function check(report, budget) {
  const problems = []
  if (report.initial.gzipKB > budget.initialKB) problems.push(`initial JS ${report.initial.gzipKB} KB > ${budget.initialKB} KB`)
  const keys = new Set(report.entries.map((entry) => entry.key))
  for (const key of [...Object.keys(budget.exempt ?? {}), ...Object.keys(budget.exceptions ?? {}), budget.defaultRoute, ...(budget.defaultRouteParts ?? [])].filter(Boolean)) {
    if (!keys.has(key)) problems.push(`${key}: named in the budget but not a lazily loaded entry of this build`)
  }
  for (const entry of report.entries) {
    if (entry.kind === "exempt") continue
    const exception = budget.exceptions?.[entry.key]
    const limit = exception?.maxKB ?? (entry.kind === "lazy" ? budget.lazyKB : budget.routeKB)
    if (entry.addedKB > limit) problems.push(`${entry.key} adds ${entry.addedKB} KB > ${limit} KB${exception ? " (its recorded exception)" : ""}`)
  }
  const parts = budget.defaultRouteParts ?? []
  const partsLimit = budget.exceptions?.[budget.defaultRoute]?.maxKB ?? budget.routeKB
  if (parts.length && report.defaultRoute && report.defaultRoute.withPartsAddedKB > partsLimit) {
    problems.push(`${budget.defaultRoute} with ${parts.join(", ")} adds ${report.defaultRoute.withPartsAddedKB} KB > ${partsLimit} KB`)
  }
  const route = budget.defaultRoute && budget.exceptions?.[budget.defaultRoute]
  if (route?.withInitialMaxKB != null && report.defaultRoute && report.defaultRoute.withInitialKB > route.withInitialMaxKB) {
    problems.push(`initial JS + ${budget.defaultRoute}: ${report.defaultRoute.withInitialKB} KB > ${route.withInitialMaxKB} KB (its recorded exception)`)
  }
  return problems
}

/**
 * A measured page load (`browser.mjs --only load`, its `scripts.files`) split by budget: the
 * initial JS, the default route (its static closure and those of `defaultRouteParts`, the lazy
 * parts it fetches as it renders), the exempt editors and everything else the view fetched
 * before it was idle (lazy parts of the first view, e.g. the model bar).
 */
export function firstView(manifest, html, size, budget, files) {
  const initial = closure(manifest, entryFiles(html))
  const route = budget.defaultRoute ? defaultView(manifest, budget) : new Set()
  const exemptFiles = new Set(Object.keys(budget.exempt ?? {}).flatMap((key) => (manifest[key] ? [...closure(manifest, [manifest[key].file])] : [])))
  const parts = { initial: 0, defaultRoute: 0, exempt: 0, other: 0 }
  for (const file of files) {
    const kb = size(file).gzip / 1024
    if (initial.has(file)) parts.initial += kb
    else if (route.has(file)) parts.defaultRoute += kb
    else if (exemptFiles.has(file)) parts.exempt += kb
    else parts.other += kb
  }
  const total = Object.values(parts).reduce((a, b) => a + b, 0)
  return { scripts: files.length, totalKB: round1(total), withoutExemptKB: round1(total - parts.exempt), ...Object.fromEntries(Object.entries(parts).map(([key, value]) => [`${key}KB`, round1(value)])) }
}
