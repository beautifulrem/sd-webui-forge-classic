// Browser measurements against a running WebUI (docs/research/ui-performance.md).
//
//   node tools/perf/browser.mjs --base http://127.0.0.1:7861 --out /tmp/remi-perf-out \
//     [--only load,inp,networks,gallery,preview,overhead] [--trace /tmp/remi-perf-trace.jsonl] \
//     [--prompt prompt300.txt] [--steps 24] [--size 768x768] [--jobs 4] [--pairs 6] [--gap 20]
//     [--headed] [--channel chrome] [--cpu-throttle 4] [--window 15] [--batch-count 1] [--repeat 1]
//     [--max-repeat <jobs; adaptive>] [--target-upper95 1] [--mute /tmp/remi-perf-mute] [--save-trace]
//     [--dist <the build the server serves; default $REMI_PERF_DIST or webui/dist>] [--inject-css "<css>"]
//
// Drives Chromium through Playwright and the DevTools protocol:
// - load: JS fetched until the Generate view is idle (gzip size of each script from webui/dist),
//   DOM elements and CDP Performance.getMetrics after a GC, for txt2img and img2img;
// - inp: types a prompt into the prompt editor (tag autocomplete on) and computes INP from
//   Event Timing entries (max duration per interactionId; interactions under the 16 ms
//   reporting threshold count as 16 ms);
// - networks / gallery: scrolls the virtualized grid from top to bottom at a fixed speed in a
//   rAF loop and records every frame interval, plus a CDP trace of the run;
// - preview: generates through the UI (infotext pasted, Generate clicked) and timestamps every
//   preview frame when the socket receives it and when its <img> is painted (wall clock, see
//   the clock check), joined with the server's trace (tools/perf/serve_traced.py);
// - overhead: alternates jobs with this page connected (previews on) and jobs sent by the API
//   with no client connected, and compares sampling it/s from the server trace;
// - interleaved: the same comparison inside one long job (see measureInterleaved).
import { execSync } from "node:child_process"
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import path from "node:path"
import { gzipSync } from "node:zlib"
import { chromium } from "@playwright/test"
import { mean, pairStats, pairsNeeded, quantile, repeatDecision, round, summary, T95, tQuantile } from "./stats.mjs"

const args = Object.fromEntries(
  process.argv.slice(2).reduce((pairs, arg, i, all) => {
    if (arg.startsWith("--")) pairs.push([arg.slice(2), all[i + 1] && !all[i + 1].startsWith("--") ? all[i + 1] : "1"])
    return pairs
  }, []),
)
const BASE = args.base ?? "http://127.0.0.1:7861"
const OUT = args.out ?? "/tmp/remi-perf-out"
const ONLY = new Set((args.only ?? "load,inp,networks,gallery,preview,overhead,interleaved").split(","))
const TRACE = args.trace ?? "/tmp/remi-perf-trace.jsonl"
const STEPS = Number(args.steps ?? 24)
const JOBS = Number(args.jobs ?? 4)
const THROTTLE = Number(args["cpu-throttle"] ?? 1)
const RESULT_FILE = path.join(OUT, `browser-${[...ONLY].join("-")}${THROTTLE > 1 ? `-x${THROTTLE}` : ""}.json`)
/** Images per job (`preview`, `interleaved`): the step starts again at 0 for each image of the job. */
const BATCH = Number(args["batch-count"] ?? 1)
// the build the server serves (serve_traced.py's REMI_PERF_DIST), for the scripts' gzip sizes
const DIST = path.resolve(args.dist ?? process.env.REMI_PERF_DIST ?? path.join(import.meta.dirname, "../../dist"))
mkdirSync(OUT, { recursive: true })

const PROMPT = "masterpiece, best quality, 1girl, silver hair, blue eyes, school uniform, cherry blossoms, upper body, smile"
const SIZE = args.size ?? "768x768"
const infotext = (steps) =>
  `${PROMPT}\nNegative prompt: worst quality, low quality\nSteps: ${steps}, Sampler: Euler, Schedule type: Simple, CFG scale: 1, Shift: 3.5, Seed: 12345, Size: ${SIZE}`

const results = { base: BASE, date: new Date().toISOString(), throttle: THROTTLE }
const log = (...parts) => console.log(...parts)

// ---------------------------------------------------------------------------- statistics

// quantile, mean, round, summary, pairStats, ...: ./stats.mjs

// ---------------------------------------------------------------------------- browser

// Event Timing, long animation frames and preview frames, collected from the first script on.
function instrument() {
  const perf = (window.__perf = { events: [], loafs: [], frames: [], paints: [] })
  new PerformanceObserver((list) => {
    for (const e of list.getEntries())
      if (e.interactionId) perf.events.push({ id: e.interactionId, name: e.name, start: e.startTime, duration: e.duration, delay: e.processingStart - e.startTime, processing: e.processingEnd - e.processingStart })
  }).observe({ type: "event", durationThreshold: 16, buffered: true })
  try {
    new PerformanceObserver((list) => {
      for (const e of list.getEntries())
        perf.loafs.push({ start: e.startTime, duration: e.duration, blocking: e.blockingDuration, scripts: e.scripts.slice(0, 3).map((s) => ({ src: s.sourceURL?.split("/").pop(), fn: s.sourceFunctionName, d: Math.round(s.duration) })) })
    }).observe({ type: "long-animation-frame", buffered: true })
  } catch {}
  const wall = () => performance.timeOrigin + performance.now()
  // Preview frames: our listener is added before the app's, so it runs first; the object URL the
  // app creates for the frame right after is tagged with the frame, and the <img> showing it is
  // timed when it has been decoded and the next frame is drawn.
  let lastFrame = null
  const NativeWS = window.WebSocket
  window.WebSocket = class extends NativeWS {
    constructor(...a) {
      super(...a)
      this.addEventListener("message", (event) => {
        if (typeof event.data === "string") return
        const received = wall()
        const view = new DataView(event.data instanceof ArrayBuffer ? event.data : new ArrayBuffer(10))
        lastFrame = { i: perf.frames.length, seq: view.getUint32(2, true), step: view.getUint32(6, true), received, bytes: event.data.byteLength ?? 0 }
        perf.frames.push(lastFrame)
      })
    }
  }
  const createObjectURL = URL.createObjectURL.bind(URL)
  const tagged = new Map()
  URL.createObjectURL = (blob) => {
    const url = createObjectURL(blob)
    if (lastFrame && blob?.type?.startsWith("image/")) {
      tagged.set(url, lastFrame)
      lastFrame = null
    }
    return url
  }
  const seen = new WeakMap()
  const watch = (img) => {
    const url = img.getAttribute("src")
    const frame = tagged.get(url)
    if (!frame || seen.get(img) === url) return
    seen.set(img, url)
    const set = wall()
    img
      .decode()
      .then(() => requestAnimationFrame(() => perf.paints.push({ i: frame.i, seq: frame.seq, step: frame.step, set, painted: wall() })))
      .catch(() => {})
  }
  new MutationObserver((records) => {
    for (const record of records) {
      if (record.type === "attributes" && record.target.tagName === "IMG") watch(record.target)
      for (const node of record.addedNodes ?? []) {
        if (node.tagName === "IMG") watch(node)
        else if (node.querySelectorAll) node.querySelectorAll("img").forEach(watch)
      }
    }
  }).observe(document, { subtree: true, childList: true, attributes: true, attributeFilter: ["src"] })
  perf.clock = () => ({ date: Date.now(), wall: wall() })
}

async function newPage(browser) {
  const context = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 })
  const page = await context.newPage()
  await page.addInitScript(instrument)
  // A/B experiments without a rebuild, e.g. --inject-css ".animate-in{animation:none!important}"
  if (args["inject-css"])
    await page.addInitScript((css) => {
      document.addEventListener("DOMContentLoaded", () => document.head.append(Object.assign(document.createElement("style"), { textContent: css })))
    }, args["inject-css"])
  const cdp = await context.newCDPSession(page)
  await cdp.send("Performance.enable")
  if (THROTTLE > 1) await cdp.send("Emulation.setCPUThrottlingRate", { rate: THROTTLE })
  page.on("pageerror", (error) => log("  pageerror:", error.message))
  return { context, page, cdp }
}

async function metrics(cdp, { gc = true } = {}) {
  if (gc) await cdp.send("HeapProfiler.collectGarbage").catch(() => {})
  const { metrics } = await cdp.send("Performance.getMetrics")
  return Object.fromEntries(metrics.map((m) => [m.name, m.value]))
}

async function openGenerate(page, tab = "txt2img") {
  await page.goto(`${BASE}/#/${tab}`)
  await page.getByRole("textbox", { name: "Prompt", exact: true }).waitFor({ timeout: 60_000 })
  await page.waitForLoadState("networkidle")
  await page.waitForTimeout(1000)
}

// ---------------------------------------------------------------------------- load + DOM

const gzipOf = new Map()
function scriptGzip(url) {
  const file = path.join(DIST, new URL(url).pathname)
  if (!existsSync(file)) {
    // not the build the server serves: the totals would be too small
    if (!gzipOf.has(file)) log(`load: WARNING ${new URL(url).pathname} is not in ${DIST} (pass --dist <the served build>)`)
    gzipOf.set(file, { raw: 0, gzip: 0 })
    return gzipOf.get(file)
  }
  if (!gzipOf.has(file)) {
    const raw = readFileSync(file)
    gzipOf.set(file, { raw: raw.length, gzip: gzipSync(raw, { level: 6 }).length })
  }
  return gzipOf.get(file)
}

async function measureLoad(browser) {
  const out = {}
  for (const tab of ["txt2img", "img2img"]) {
    const { context, page, cdp } = await newPage(browser)
    const scripts = []
    page.on("response", (response) => {
      if (response.request().resourceType() === "script") scripts.push(response.url())
    })
    const started = Date.now()
    await openGenerate(page, tab)
    const ready = Date.now() - started
    const nav = await page.evaluate(() => {
      const [n] = performance.getEntriesByType("navigation")
      const fcp = performance.getEntriesByName("first-contentful-paint")[0]
      const js = performance.getEntriesByType("resource").filter((r) => r.name.endsWith(".js"))
      const css = performance.getEntriesByType("resource").filter((r) => r.name.endsWith(".css"))
      const api = performance.getEntriesByType("resource").filter((r) => /\/(api|sdapi)\//.test(r.name))
      const bytes = (list, key) => list.reduce((sum, r) => sum + r[key], 0)
      return {
        domContentLoaded: n.domContentLoadedEventEnd,
        load: n.loadEventEnd,
        fcp: fcp?.startTime,
        // bytes on the wire (encoded) and inflated, as the browser saw them
        transfer: { jsEncoded: bytes(js, "encodedBodySize"), jsDecoded: bytes(js, "decodedBodySize"), cssEncoded: bytes(css, "encodedBodySize"), apiEncoded: bytes(api, "encodedBodySize"), apiDecoded: bytes(api, "decodedBodySize"), apiRequests: api.length },
      }
    })
    const dom = await page.evaluate(() => ({ elements: document.getElementsByTagName("*").length, textNodes: document.createTreeWalker(document, NodeFilter.SHOW_TEXT).nextNode ? (() => { let n = 0; const w = document.createTreeWalker(document, NodeFilter.SHOW_TEXT); while (w.nextNode()) n++; return n })() : 0, depth: (() => { let max = 0; for (const el of document.getElementsByTagName("*")) { let d = 0; for (let p = el; p; p = p.parentElement) d++; max = Math.max(max, d) } return max })() }))
    const m = await metrics(cdp)
    const js = scripts.map((url) => ({ file: new URL(url).pathname, ...scriptGzip(url) }))
    // Each collapsed section opened on its own (then closed again): the heaviest single section.
    const sections = await page.evaluate(async () => {
      const count = () => document.getElementsByTagName("*").length
      const base = count()
      const buttons = [...document.querySelectorAll('main button[aria-expanded="false"]')].filter((el) => !el.hasAttribute("aria-haspopup"))
      const rows = []
      for (const button of buttons) {
        if (!button.isConnected) continue
        button.click()
        await new Promise((r) => setTimeout(r, 250))
        rows.push({ label: (button.getAttribute("aria-label") || button.textContent || "").trim().slice(0, 60), elements: count() - base })
        button.click()
        await new Promise((r) => setTimeout(r, 150))
      }
      rows.sort((a, b) => b.elements - a.elements)
      return { base, sections: rows.length, heaviest: rows.slice(0, 8), after: count() }
    })
    // Expand every collapsed section once to show the worst case.
    const expanded = await page.evaluate(async () => {
      let clicks = 0
      for (let round = 0; round < 4; round++) {
        const closed = [...document.querySelectorAll('main [aria-expanded="false"]')].filter((el) => el.tagName === "BUTTON" && !el.closest('[role="menu"],[role="listbox"],[aria-haspopup]') && !el.hasAttribute("aria-haspopup"))
        if (!closed.length) break
        for (const el of closed) {
          el.click()
          clicks++
        }
        await new Promise((r) => setTimeout(r, 400))
      }
      return { clicks, elements: document.getElementsByTagName("*").length }
    })
    out[tab] = {
      readyMs: ready,
      navigation: nav,
      scripts: { count: js.length, gzipKB: round(js.reduce((n, s) => n + s.gzip, 0) / 1024), rawKB: round(js.reduce((n, s) => n + s.raw, 0) / 1024), files: js },
      dom: { ...dom, cdpNodes: m.Nodes, documents: m.Documents, jsHeapMB: round(m.JSHeapUsedSize / 1048576), layoutCount: m.LayoutCount, recalcStyleCount: m.RecalcStyleCount },
      singleSection: sections,
      allSectionsExpanded: expanded,
    }
    log(`load ${tab}: ${js.length} scripts ${out[tab].scripts.gzipKB} KB gzip (${round(nav.transfer.jsEncoded / 1024)} KB on the wire, API ${round(nav.transfer.apiEncoded / 1024)} KB); ${dom.elements} elements (${m.Nodes} CDP nodes), expanded: ${expanded.elements} elements; ready ${ready} ms, FCP ${round(nav.fcp)} ms`)
    await context.close()
  }
  results.load = out
}

// ---------------------------------------------------------------------------- INP

async function measureInp(browser) {
  const text = readFileSync(args.prompt ?? path.join(OUT, "prompt300.txt"), "utf8").trim()
  const delay = Number(args["type-delay"] ?? 100)
  const { context, page, cdp } = await newPage(browser)
  await openGenerate(page)
  const prompt = page.getByRole("textbox", { name: "Prompt", exact: true })
  await prompt.click()
  await page.keyboard.press("Meta+A")
  await page.keyboard.press("Backspace")
  await page.waitForTimeout(500)
  await page.evaluate(() => {
    window.__perf.events.length = 0
    window.__perf.loafs.length = 0
    // count autocomplete popups (CodeMirror's tooltip) as they open
    window.__perf.popups = 0
    new MutationObserver((records) => {
      for (const r of records) for (const n of r.addedNodes) if (n.classList?.contains("cm-tooltip-autocomplete")) window.__perf.popups++
    }).observe(document.body, { subtree: true, childList: true })
  })
  let completions = 0
  page.on("request", (request) => {
    if (/prompt-tools\/complete|complete\?/.test(request.url())) completions++
  })
  const before = await metrics(cdp)
  const started = Date.now()
  for (const ch of text) {
    await page.keyboard.type(ch)
    await page.waitForTimeout(delay)
  }
  await page.waitForTimeout(1000)
  const elapsed = Date.now() - started
  const after = await metrics(cdp)
  const data = await page.evaluate(() => ({ events: window.__perf.events, loafs: window.__perf.loafs, popups: window.__perf.popups }))
  const popups = data.popups
  const typed = await prompt.textContent()
  const byId = new Map()
  for (const e of data.events) byId.set(e.id, Math.max(byId.get(e.id) ?? 0, e.duration))
  // keydown + keyup of one key share an interactionId; every typed character is one interaction
  const interactions = text.length
  const reported = [...byId.values()]
  const all = [...reported, ...Array(Math.max(0, interactions - reported.length)).fill(16)]
  // web-vitals: the worst interaction, ignoring one per 50 interactions
  const sorted = [...all].sort((a, b) => b - a)
  const inp = sorted[Math.min(sorted.length - 1, Math.floor(all.length / 50))]
  results.inp = {
    chars: text.length,
    typeDelayMs: delay,
    elapsedMs: elapsed,
    typedMatches: typed?.replace(/\s+/g, " ").trim() === text.replace(/\s+/g, " "),
    autocompletePopups: popups,
    completionRequests: completions,
    interactions,
    reportedOver16ms: reported.length,
    p50: round(quantile(all, 0.5)),
    p95: round(quantile(all, 0.95)),
    p99: round(quantile(all, 0.99)),
    max: round(Math.max(...all)),
    inpWebVitals: inp,
    worst: data.events.sort((a, b) => b.duration - a.duration).slice(0, 8),
    longAnimationFrames: { count: data.loafs.length, durations: summary(data.loafs.map((l) => l.duration)), worst: data.loafs.sort((a, b) => b.duration - a.duration).slice(0, 5) },
    taskDurationS: round(after.TaskDuration - before.TaskDuration, 2),
    scriptDurationS: round(after.ScriptDuration - before.ScriptDuration, 2),
  }
  log(`inp: ${interactions} keys, ${reported.length} over 16 ms; p95 ${results.inp.p95} ms, INP ${inp} ms, max ${results.inp.max} ms; autocomplete popups ${popups}, requests ${completions}; text ok ${results.inp.typedMatches}`)
  await context.close()
}

// ---------------------------------------------------------------------------- scrolling

const TRACE_CATEGORIES = ["devtools.timeline", "disabled-by-default-devtools.timeline.frame", "benchmark", "latencyInfo", "rail"]

/**
 * What the renderer's main thread did during each frame interval longer than `limitMs` (the
 * page's own rAF timestamps, put on the trace's clock by a `console.timeStamp` anchor): the
 * outermost timeline events in the interval by name, and how much of it they cover. A long frame
 * with less than a frame's worth of main-thread work was not caused by the page's scripts, layout
 * or GC (the renderer was not scheduled, or the compositor / GPU was late).
 */
function longFrames(traceEvents, frames, limitMs = 20) {
  const out = { count: 0, pageWork: 0, noPageWork: 0, byEvent: {}, worst: [] }
  const anchor = traceEvents.find((e) => e.name === "TimeStamp" && e.args?.data?.message === "perf-anchor")
  if (!anchor || frames.anchor == null) return { ...out, error: "no anchor in the trace" }
  const offset = anchor.ts - frames.anchor * 1000 // trace µs = page ms * 1000 + offset
  const key = `${anchor.pid}:${anchor.tid}` // the page's main thread called console.timeStamp
  const work = traceEvents.filter((e) => e.ph === "X" && e.dur > 0 && `${e.pid}:${e.tid}` === key).sort((a, b) => a.ts - b.ts || b.dur - a.dur)
  let w = 0
  for (let i = 1; i < frames.times.length; i++) {
    const gapMs = frames.times[i] - frames.times[i - 1]
    if (gapMs <= limitMs) continue
    out.count++
    const from = frames.times[i - 1] * 1000 + offset
    const to = frames.times[i] * 1000 + offset
    while (w < work.length && work[w].ts + work[w].dur < from) w++
    const byName = {}
    let busy = 0
    let coveredTo = from
    for (let j = w; j < work.length && work[j].ts < to; j++) {
      const e = work[j]
      const start = Math.max(e.ts, coveredTo)
      const end = Math.min(e.ts + e.dur, to)
      if (end <= start) continue // inside an event already counted
      busy += end - start
      byName[e.name] = (byName[e.name] ?? 0) + (end - start) / 1000
      coveredTo = end
    }
    const busyMs = busy / 1000
    if (busyMs > 16.7) out.pageWork++
    else out.noPageWork++
    for (const [name, ms] of Object.entries(byName)) out.byEvent[name] = round((out.byEvent[name] ?? 0) + ms, 1)
    out.worst.push({ gapMs: round(gapMs, 1), mainBusyMs: round(busyMs, 1), top: Object.entries(byName).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([name, ms]) => [name, round(ms, 1)]) })
  }
  out.worst = out.worst.sort((a, b) => b.gapMs - a.gapMs).slice(0, 10)
  return out
}

async function scrollRun(page, cdp, scroller, pxPerFrame, label) {
  const tracePath = path.join(OUT, `trace-${label}.json`)
  const chunks = []
  cdp.on("Tracing.dataCollected", ({ value }) => chunks.push(...value))
  const done = new Promise((resolve) => cdp.once("Tracing.tracingComplete", resolve))
  await cdp.send("Tracing.start", { traceConfig: { includedCategories: TRACE_CATEGORIES }, transferMode: "ReportEvents" })
  const before = await metrics(cdp)
  const frames = await page.evaluate(
    ([selector, step]) =>
      new Promise((resolve) => {
        const el = document.querySelector(selector)
        el.scrollTop = 0
        const times = []
        const start = performance.now()
        // puts the rAF times on the trace's clock (longFrames)
        const anchor = performance.now()
        console.timeStamp("perf-anchor")
        const tick = (ts) => {
          times.push(ts)
          el.scrollTop += step
          const atEnd = el.scrollTop + el.clientHeight >= el.scrollHeight - 2
          if (atEnd || ts - start > 100_000) resolve({ times, anchor, scrollHeight: el.scrollHeight, rendered: el.querySelectorAll("img").length })
          else requestAnimationFrame(tick)
        }
        requestAnimationFrame(tick)
      }),
    [scroller, pxPerFrame],
  )
  const after = await metrics(cdp)
  await cdp.send("Tracing.end")
  await done
  writeFileSync(tracePath, JSON.stringify({ traceEvents: chunks }))
  const intervals = frames.times.slice(1).map((t, i) => t - frames.times[i])
  const elapsed = (frames.times.at(-1) - frames.times[0]) / 1000
  // Frames the compositor reports (PipelineReporter): presented vs dropped.
  const states = {}
  for (const e of chunks) {
    const reporter = e.name === "PipelineReporter" && e.ph === "b" ? (e.args?.frame_reporter ?? e.args?.chrome_frame_reporter) : null
    if (reporter) states[reporter.state] = (states[reporter.state] ?? 0) + 1
  }
  const out = {
    pxPerFrame,
    frames: frames.times.length,
    seconds: round(elapsed, 2),
    fps: round((frames.times.length - 1) / elapsed),
    interval: summary(intervals),
    over20ms: intervals.filter((d) => d > 20).length,
    over33ms: intervals.filter((d) => d > 33.4).length,
    scrollHeight: frames.scrollHeight,
    pipeline: states,
    taskDurationS: round(after.TaskDuration - before.TaskDuration, 2),
    layoutDurationS: round(after.LayoutDuration - before.LayoutDuration, 2),
    recalcStyleDurationS: round(after.RecalcStyleDuration - before.RecalcStyleDuration, 2),
    nodesAfter: after.Nodes,
    longFrames: longFrames(chunks, frames),
    trace: tracePath,
  }
  log(`  ${label} @${pxPerFrame}px/frame: ${out.fps} fps over ${out.seconds}s, p95 ${out.interval.p95} ms, >20ms ${out.over20ms}/${intervals.length}, pipeline ${JSON.stringify(states)}; long frames with page work ${out.longFrames.pageWork}, without ${out.longFrames.noPageWork}`)
  return out
}

async function idleFps(page) {
  return page.evaluate(
    () =>
      new Promise((resolve) => {
        const times = []
        const tick = (ts) => {
          times.push(ts)
          if (times.length < 121) requestAnimationFrame(tick)
          else resolve(Math.round((120 / (times.at(-1) - times[0])) * 1000 * 10) / 10)
        }
        requestAnimationFrame(tick)
      }),
  )
}

async function measureNetworks(browser) {
  const { context, page, cdp } = await newPage(browser)
  await page.goto(`${BASE}/#/networks`)
  const grid = page.getByRole("listbox").first()
  await grid.waitFor({ timeout: 60_000 })
  await page.waitForLoadState("networkidle")
  const total = await page.evaluate(async (base) => (await (await fetch(`${base}/api/v2/networks/lora?limit=1`)).json()).total, BASE)
  await page.evaluate(() => document.querySelector('[role="listbox"][aria-multiselectable]')?.setAttribute("data-perf-scroller", ""))
  const idle = await idleFps(page)
  const out = { total, idleFps: idle, runs: [] }
  for (const px of [40, 160]) out.runs.push(await scrollRun(page, cdp, "[data-perf-scroller]", px, `networks-${px}`))
  // second pass: everything was visited once (thumbnails cached)
  out.runs.push(await scrollRun(page, cdp, "[data-perf-scroller]", 40, "networks-40-warm"))
  out.dom = await page.evaluate(() => document.getElementsByTagName("*").length)
  results.networks = out
  await context.close()
}

async function measureGallery(browser) {
  const { context, page, cdp } = await newPage(browser)
  await page.goto(`${BASE}/#/gallery`)
  await page.waitForLoadState("networkidle")
  await page.waitForTimeout(1500)
  const selector = await page.evaluate(() => {
    let best = null
    for (const el of document.querySelectorAll("main *")) {
      const style = getComputedStyle(el)
      if (/(auto|scroll)/.test(style.overflowY) && el.scrollHeight > el.clientHeight + 100 && (!best || el.scrollHeight > best.scrollHeight)) best = el
    }
    best?.setAttribute("data-perf-scroller", "")
    return best ? "[data-perf-scroller]" : null
  })
  if (!selector) throw new Error("gallery scroller not found")
  // Load every page first (the grid fetches the next page as its end comes into view).
  const loadStart = Date.now()
  let count = 0
  for (let i = 0; i < 400; i++) {
    count = await page.evaluate(() => {
      const el = document.querySelector("[data-perf-scroller]")
      el.scrollTop = el.scrollHeight
      return el.scrollHeight
    })
    await page.waitForTimeout(150)
    const done = await page.evaluate(() => {
      const el = document.querySelector("[data-perf-scroller]")
      return el.scrollHeight === Number(el.dataset.lastHeight ?? -1) ? Number(el.dataset.same ?? 0) + 1 : ((el.dataset.lastHeight = el.scrollHeight), 0)
    })
    await page.evaluate((same) => (document.querySelector("[data-perf-scroller]").dataset.same = same), done)
    if (done >= 12) break
  }
  const loaded = await page.evaluate(() => {
    const text = document.querySelector("main")?.innerText ?? ""
    return (text.match(/([\d,]+)\s+(?:images|items)/i) ?? [])[1] ?? null
  })
  const idle = await idleFps(page)
  const out = { loadAllMs: Date.now() - loadStart, scrollHeight: count, countText: loaded, idleFps: idle, runs: [] }
  for (const px of [40, 160]) out.runs.push(await scrollRun(page, cdp, selector, px, `gallery-${px}`))
  out.runs.push(await scrollRun(page, cdp, selector, 40, "gallery-40-warm"))
  out.dom = await page.evaluate(() => document.getElementsByTagName("*").length)
  results.gallery = out
  await context.close()
}

// ---------------------------------------------------------------------------- generation

function readTrace(since) {
  if (!existsSync(TRACE)) return []
  return readFileSync(TRACE, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line))
    .filter((e) => (e.t ?? 0) * 1000 >= since)
}

async function waitIdle(timeoutMs = 900_000) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const queue = await (await fetch(`${BASE}/api/v2/queue`)).json().catch(() => null)
    const running = queue?.running ?? queue?.current ?? null
    const pending = queue?.items?.length ?? queue?.pending?.length ?? queue?.size ?? 0
    if (!running && !pending) return
    await new Promise((r) => setTimeout(r, 500))
  }
  throw new Error("queue still busy")
}

async function waitJob(jobId) {
  for (;;) {
    const job = await (await fetch(`${BASE}/api/v2/jobs/${jobId}`)).json()
    if (!["queued", "running"].includes(job.status)) return job
    await new Promise((r) => setTimeout(r, 300))
  }
}

/** Steps per second of the sampling loop between two wall times (the first step excluded). */
function itsFrom(trace, from, to) {
  const steps = trace.filter((e) => e.e === "step" && e.t * 1000 >= from && e.t * 1000 <= to)
  if (steps.length < 3) return null
  const deltas = steps.slice(1).map((s, i) => s.t - steps[i].t)
  return { steps: steps.length, its: round((steps.length - 1) / (steps.at(-1).t - steps[0].t), 4), stepMs: summary(deltas.map((d) => d * 1000)) }
}

async function uiJob(page, steps, { batchCount = 1 } = {}) {
  const prompt = page.getByRole("textbox", { name: "Prompt", exact: true })
  await prompt.click()
  await page.keyboard.press("Meta+A")
  await page.keyboard.insertText(infotext(steps))
  await page.getByRole("button", { name: /^Read parameters from the prompt/ }).click()
  await page.waitForTimeout(300)
  // more images (steps) in the one job the page owns: the form's request with another batch count
  const batch = (route) => {
    const request = route.request()
    if (request.method() !== "POST") return route.continue()
    const body = request.postDataJSON()
    return route.continue({ postData: JSON.stringify({ ...body, core: { ...body.core, batch_count: batchCount } }) })
  }
  if (batchCount > 1) await page.route("**/api/v2/jobs", batch)
  const posted = page.waitForResponse((r) => r.url().endsWith("/api/v2/jobs") && r.request().method() === "POST")
  const start = Date.now()
  await page.getByRole("button", { name: /^Generate/ }).first().click()
  const response = await posted
  if (batchCount > 1) await page.unroute("**/api/v2/jobs", batch)
  const body = response.request().postDataJSON()
  if (!response.ok()) throw new Error(`POST /api/v2/jobs: ${response.status()} ${await response.text()}`)
  const { job_id: jobId } = await response.json()
  const job = await waitJob(jobId)
  return { jobId, body, start, end: Date.now(), status: job.status }
}

async function apiJob(body) {
  const start = Date.now()
  const response = await fetch(`${BASE}/api/v2/jobs`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
  if (!response.ok) throw new Error(`POST /api/v2/jobs: ${response.status} ${await response.text()}`)
  const { job_id: jobId } = await response.json()
  const job = await waitJob(jobId)
  return { jobId, start, end: Date.now(), status: job.status }
}

/** Renderer main-thread metrics (CDP) and the CPU time of every Chromium process (ps), now. */
async function pageCounters(cdp) {
  return { metrics: await metrics(cdp, { gc: false }), cpu: treeCpu(process.pid, { includeRoot: false }), at: Date.now() }
}

/** What the page cost between two counters, per step (and per second while idle). */
function pageCost(a, b, steps, idle = null) {
  const per = (value) => value / Math.max(1, steps)
  const seconds = (b.at - a.at) / 1000
  const cost = {
    steps,
    seconds: round(seconds, 1),
    taskMs: round(per((b.metrics.TaskDuration - a.metrics.TaskDuration) * 1000), 2),
    scriptMs: round(per((b.metrics.ScriptDuration - a.metrics.ScriptDuration) * 1000), 2),
    layoutMs: round(per((b.metrics.LayoutDuration - a.metrics.LayoutDuration) * 1000), 2),
    recalcStyleMs: round(per((b.metrics.RecalcStyleDuration - a.metrics.RecalcStyleDuration) * 1000), 2),
    layouts: round(per(b.metrics.LayoutCount - a.metrics.LayoutCount), 2),
    chromiumCpuMs: round(per((b.cpu - a.cpu) * 1000), 1),
  }
  if (idle) {
    // over what the same open page costs while nothing runs
    const stepSeconds = seconds / Math.max(1, steps)
    cost.overIdle = { taskMs: round(cost.taskMs - idle.taskMs * stepSeconds, 2), chromiumCpuMs: round(cost.chromiumCpuMs - idle.chromiumCpuMs * stepSeconds, 1) }
  }
  return cost
}

async function measurePreview(browser) {
  await waitIdle()
  const previewFrom = Date.now()
  const { context, page, cdp } = await newPage(browser)
  await openGenerate(page)
  const clock = await page.evaluate(() => window.__perf.clock())
  // the open page's own cost while nothing runs (per second), the reference for the per-step cost
  const idleFrom = await pageCounters(cdp)
  await page.waitForTimeout(10_000)
  const idle = pageCost(idleFrom, await pageCounters(cdp), 10)
  const jobs = []
  // compositor frames the page draws while the first job samples (from its second step to its last)
  const chunks = []
  cdp.on("Tracing.dataCollected", ({ value }) => chunks.push(...value))
  const traced = new Promise((resolve) => cdp.once("Tracing.tracingComplete", resolve))
  const submitted = Date.now()
  const jobSteps = STEPS * BATCH
  const first = uiJob(page, STEPS, { batchCount: BATCH })
  const stepsSince = (t) => readTrace(t).filter((e) => e.e === "step")
  while (stepsSince(submitted).length < 2) await page.waitForTimeout(100)
  const tracedFrom = Date.now()
  await cdp.send("Tracing.start", { traceConfig: { includedCategories: TRACE_CATEGORIES }, transferMode: "ReportEvents" })
  while (stepsSince(submitted).length < jobSteps) await page.waitForTimeout(100)
  await cdp.send("Tracing.end")
  await traced
  const tracedSteps = stepsSince(tracedFrom).length
  jobs.push(await first)
  if (args["save-trace"]) writeFileSync(path.join(OUT, "trace-preview.json"), JSON.stringify({ traceEvents: chunks }))
  const reports = chunks.filter((e) => e.name === "PipelineReporter" && e.ph === "b")
  const presentedReports = reports.filter((e) => /PRESENTED/.test(e.args?.frame_reporter?.state ?? ""))
  const presented = presentedReports.length
  // a frame can be reported twice (fully presented, and "partial": without that frame's main-thread update)
  const sequences = new Set(presentedReports.map((e) => e.args?.frame_reporter?.frame_sequence ?? e.id))
  const states = {}
  for (const e of reports) {
    const state = e.args?.frame_reporter?.state ?? "?"
    states[state] = (states[state] ?? 0) + 1
  }
  const firstJobSteps = Math.max(1, tracedSteps)
  // the page's cost per step over a job that is not traced (tracing costs CPU itself)
  let cost = null
  for (let i = 1; i < JOBS; i++) {
    const submittedAt = Date.now()
    const next = uiJob(page, STEPS, { batchCount: BATCH })
    if (i === 1) {
      while (stepsSince(submittedAt).length < 2) await page.waitForTimeout(100)
      const from = await pageCounters(cdp)
      const fromSteps = stepsSince(submittedAt).length
      while (stepsSince(submittedAt).length < jobSteps) await page.waitForTimeout(100)
      const to = await pageCounters(cdp)
      cost = pageCost(from, to, stepsSince(submittedAt).length - fromSteps, idle)
    }
    jobs.push(await next)
  }
  const data = await page.evaluate(() => ({ frames: window.__perf.frames, paints: window.__perf.paints }))
  await context.close()
  const since = jobs[0].start - 1000
  const trace = readTrace(since)
  const published = trace.filter((e) => e.e === "frame")
  const rows = []
  for (const paint of data.paints) {
    const frame = data.frames[paint.i]
    // the newest frame of this job and step the server published before the socket received it
    const s = published.filter((e) => e.seq === frame.seq && e.step === frame.step && e.t * 1000 <= frame.received + 5).at(-1)
    if (!s) continue
    rows.push({
      seq: paint.seq,
      step: paint.step,
      imageToPublish: (s.t - (s.image_t ?? s.t0)) * 1000,
      encode: (s.t - s.t0) * 1000,
      publishToReceive: frame.received - s.t * 1000,
      receiveToPaint: paint.painted - frame.received,
      total: paint.painted - (s.image_t ?? s.t0) * 1000,
      encodedToPaint: paint.painted - s.t * 1000,
    })
  }
  const pick = (key) => summary(rows.map((r) => r[key]))
  // frames received and painted per image of each job (a new image: the step goes back within the job)
  const images = []
  data.frames.forEach((frame, i) => {
    const previous = data.frames[i - 1]
    if (!previous || previous.seq !== frame.seq || frame.step < previous.step) images.push({ seq: frame.seq, received: 0, painted: 0 })
    images.at(-1).received++
    frame.image = images.length - 1
  })
  for (const paint of data.paints) images[data.frames[paint.i].image].painted++
  results.preview = {
    previewClients: previewClients(previewFrom, "preview"),
    jobs: jobs.map(({ jobId, start, end, status }) => ({ jobId, ms: end - start, status })),
    steps: STEPS,
    batchCount: BATCH,
    perImage: images,
    paintedShare: round(data.paints.length / Math.max(1, data.frames.length), 3),
    framesPerStep: round(sequences.size / Math.max(1, firstJobSteps), 2),
    frameReportsPerStep: round(presented / Math.max(1, firstJobSteps), 2),
    frameStates: states,
    pageIdlePerSecond: idle,
    pageCostPerStep: cost,
    framesPresentedFirstJob: sequences.size,
    clockOffsetMs: round(clock.date - clock.wall, 2),
    framesReceived: data.frames.length,
    framesPainted: data.paints.length,
    framesPublished: trace.filter((e) => e.e === "frame").length,
    imagesAssigned: trace.filter((e) => e.e === "image").length,
    matched: rows.length,
    total: pick("total"),
    imageToPublish: pick("imageToPublish"),
    encode: pick("encode"),
    publishToReceive: pick("publishToReceive"),
    receiveToPaint: pick("receiveToPaint"),
    encodedToPaint: pick("encodedToPaint"),
    its: jobs.map((j) => itsFrom(trace, j.start, j.end)),
  }
  writeFileSync(path.join(OUT, "preview-rows.json"), JSON.stringify(rows, null, 1))
  const unpainted = images.filter((image) => image.painted < image.received / 2).length
  if (unpainted) log(`preview: WARNING ${unpainted} of ${images.length} images had fewer than half of their frames painted`)
  log(`preview: ${rows.length} frames matched (${data.frames.length} received, ${data.paints.length} painted, ${images.length} images); total p50 ${results.preview.total.p50} p95 ${results.preview.total.p95} ms; image→publish p95 ${results.preview.imageToPublish.p95}, publish→receive p95 ${results.preview.publishToReceive.p95}, receive→paint p95 ${results.preview.receiveToPaint.p95}; clock offset ${results.preview.clockOffsetMs} ms; ${results.preview.framesPerStep} frames drawn per step (${results.preview.frameReportsPerStep} reports); per step: renderer tasks ${cost?.taskMs} ms, ${cost?.layouts} layouts, Chromium CPU ${cost?.chromiumCpuMs} ms (over idle ${cost?.overIdle?.chromiumCpuMs} ms)`)
}

/** CPU seconds (user + system) of a process and its descendants, from `ps`. */
function treeCpu(rootPid, { includeRoot = true } = {}) {
  const rows = execSync("ps -A -o pid=,ppid=,time=").toString().trim().split("\n").map((line) => line.trim().split(/\s+/))
  const children = new Map()
  const cpu = new Map()
  for (const [pid, ppid, time] of rows) {
    const parts = time.split(/[:-]/).map(Number)
    const seconds = parts.reduce((total, part) => total * 60 + part, 0)
    cpu.set(Number(pid), seconds)
    if (!children.has(Number(ppid))) children.set(Number(ppid), [])
    children.get(Number(ppid)).push(Number(pid))
  }
  let total = includeRoot ? (cpu.get(rootPid) ?? 0) : 0
  const stack = [...(children.get(rootPid) ?? [])]
  while (stack.length) {
    const pid = stack.pop()
    total += cpu.get(pid) ?? 0
    stack.push(...(children.get(pid) ?? []))
  }
  return total
}

/**
 * The most WebSocket clients a preview went to since `since` (serve_traced.py's "fan" records).
 * More than one: another tab of the WebUI (or a script) received the previews too, and its cost
 * counted as this page's.
 */
function previewClients(since, scenario) {
  const counts = readTrace(since).filter((e) => e.e === "fan").map((e) => e.clients)
  const most = counts.length ? Math.max(...counts) : null
  if (most > 1) log(`${scenario}: WARNING previews went to ${most} WebSocket clients, not only this page: close other tabs of the WebUI (or use another port) and measure again`)
  return most
}

async function measureOverhead(browser) {
  await waitIdle()
  const overheadFrom = Date.now()
  const pairs = Number(args.pairs ?? 6)
  const gap = Number(args.gap ?? 20) * 1000 // the same idle time before every job (the GPU cools down alike)
  const serverPid = Number(execSync("pgrep -f 'serve_traced.py|launch.py' | head -1").toString().trim()) || null
  const runs = []
  // warm-up job (model resident, kernels compiled), not counted; its request is replayed by the API jobs
  let body
  {
    const { context, page } = await newPage(browser)
    await openGenerate(page)
    body = (await uiJob(page, STEPS)).body
    await context.close()
  }
  let lastEnd = Date.now()
  const untilGap = () => new Promise((r) => setTimeout(r, Math.max(0, lastEnd + gap - Date.now())))
  for (let i = 0; i < pairs; i++) {
    for (const mode of i % 2 ? ["api", "browser"] : ["browser", "api"]) {
      let job
      let browserCpu = null
      const server0 = serverPid ? treeCpu(serverPid) : null
      if (mode === "browser") {
        // the page has been open (and idle) for a while when the job starts, as when a user clicks Generate
        const { context, page } = await newPage(browser)
        await openGenerate(page)
        await untilGap()
        const cpu0 = treeCpu(process.pid, { includeRoot: false })
        job = await uiJob(page, STEPS)
        browserCpu = treeCpu(process.pid, { includeRoot: false }) - cpu0
        job.painted = await page.evaluate(() => window.__perf.paints.length)
        await context.close()
      } else {
        await untilGap()
        job = await apiJob(body)
      }
      lastEnd = Date.now()
      const its = itsFrom(readTrace(job.start - 1000), job.start, job.end)
      const seconds = (job.end - job.start) / 1000
      runs.push({ pair: i, mode, jobId: job.jobId, ms: job.end - job.start, painted: job.painted, its: its?.its, stepMs: its?.stepMs, browserCpuPct: browserCpu == null ? null : round((browserCpu / seconds) * 100, 1), serverCpuPct: server0 == null ? null : round(((treeCpu(serverPid) - server0) / seconds) * 100, 1) })
      log(`  ${mode}: ${its?.its} it/s (${its?.steps} steps, ${job.end - job.start} ms${job.painted != null ? `, ${job.painted} previews painted, browser CPU ${runs.at(-1).browserCpuPct} %` : ""}, server CPU ${runs.at(-1).serverCpuPct} %)`)
    }
  }
  const of = (mode) => runs.filter((r) => r.mode === mode).map((r) => r.its)
  const browserIts = of("browser")
  const apiIts = of("api")
  // per pair (adjacent jobs share the thermal state): 1 - browser / api
  const perPair = [...Array(pairs).keys()].map((i) => {
    const b = runs.find((r) => r.pair === i && r.mode === "browser").its
    const a = runs.find((r) => r.pair === i && r.mode === "api").its
    return (1 - b / a) * 100
  })
  const sd = Math.sqrt(perPair.reduce((n, v) => n + (v - mean(perPair)) ** 2, 0) / Math.max(1, perPair.length - 1))
  const ci = (T95[perPair.length - 2] ?? 2) * (sd / Math.sqrt(perPair.length))
  results.overhead = {
    previewClients: previewClients(overheadFrom, "overhead"),
    steps: STEPS,
    pairs,
    gapS: gap / 1000,
    runs,
    browserIts: summary(browserIts.map((v) => v * 1000)),
    apiIts: summary(apiIts.map((v) => v * 1000)),
    overheadPct: round((1 - mean(browserIts) / mean(apiIts)) * 100, 2),
    perPairPct: perPair.map((v) => round(v, 2)),
    perPairMeanPct: round(mean(perPair), 2),
    perPairCi95Pct: round(ci, 2),
    browserCpuPct: summary(runs.filter((r) => r.browserCpuPct != null).map((r) => r.browserCpuPct)),
    serverCpuPct: { browser: summary(runs.filter((r) => r.mode === "browser" && r.serverCpuPct != null).map((r) => r.serverCpuPct)), api: summary(runs.filter((r) => r.mode === "api" && r.serverCpuPct != null).map((r) => r.serverCpuPct)) },
  }
  log(`overhead: browser ${round(mean(browserIts), 4)} it/s vs api ${round(mean(apiIts), 4)} it/s -> ${results.overhead.overheadPct} %; per pair ${results.overhead.perPairMeanPct} ± ${results.overhead.perPairCi95Pct} % (95 % CI)`)
}

/**
 * The same overhead inside one long job (no thermal drift between the compared samples): the
 * page that started the job stays open and the server is switched between serving it (previews
 * encoded, sent and painted) and acting as if no client were connected (serve_traced.py's mute
 * file), in windows of `--window` seconds in A B B A order. Every step interval that lies inside
 * one window (the first after a switch excluded) counts for that window's condition; each
 * adjacent on/off window pair gives one relative difference. `--repeat N` runs N such jobs one
 * after the other (each from a fresh page, batch count 1 if wanted) and pools their window pairs:
 * the form caps a job at 150 steps, too few pairs for a tight bound from one job. With
 * `--max-repeat M` (> N) the run is adaptive: after the N jobs it adds jobs until the pooled
 * one-sided 95 % bound settles the budget (`--target-upper95`, default 1 %: upper bound at or
 * under it, or lower bound over it) or M jobs ran (`verdict`: within / over / inconclusive). The
 * server's own cost per step, the primary overhead figure, comes from the same run's trace:
 * tools/perf/step_segments.py --run <this result file> (the run records `traceSince` / `traceUntil`).
 */
async function interleavedJob(browser, { mute, windowMs, steps, batchCount }) {
  rmSync(mute, { force: true })
  const { context, page } = await newPage(browser)
  await openGenerate(page)
  await page.waitForTimeout(5000)
  const cpu0 = treeCpu(process.pid, { includeRoot: false })
  const job = uiJob(page, steps, { batchCount })
  const windows = []
  const order = ["on", "off", "off", "on"]
  let done = false
  job.then(() => (done = true))
  // wait for the first step, then alternate
  // oxlint-disable-next-line eslint/no-unmodified-loop-condition -- `done` is set by the job promise while this awaits
  for (let i = 0; !done && i < 600; i++) {
    if (readTrace(Date.now() - 5000).some((e) => e.e === "step")) break
    await page.waitForTimeout(500)
  }
  // oxlint-disable-next-line eslint/no-unmodified-loop-condition -- `done` is set by the job promise while this awaits
  for (let i = 0; !done; i++) {
    const condition = order[i % order.length]
    if (condition === "off") writeFileSync(mute, "")
    else rmSync(mute, { force: true })
    const start = Date.now()
    // oxlint-disable-next-line eslint/no-unmodified-loop-condition -- `done` is set by the job promise while this awaits
    while (!done && Date.now() - start < windowMs) await page.waitForTimeout(100)
    windows.push({ condition, start, end: Date.now() })
  }
  rmSync(mute, { force: true })
  const finished = await job
  const browserCpu = treeCpu(process.pid, { includeRoot: false }) - cpu0
  const { painted, received } = await page.evaluate(() => ({ painted: window.__perf.paints.length, received: window.__perf.frames.length }))
  await context.close()
  const trace = readTrace(windows[0].start - 1000).filter((e) => e.e === "step")
  for (const w of windows) w.steps = []
  for (let i = 1; i < trace.length; i++) {
    // consecutive sampler steps only (not across the images of a batch: decoding and saving in between)
    if (trace[i].step !== trace[i - 1].step + 1) continue
    const a = trace[i - 1].t * 1000
    const z = trace[i].t * 1000
    const w = windows.find((w) => a >= w.start && z <= w.end)
    if (w) w.steps.push(z - a)
  }
  return { windows, painted, received, jobMs: finished.end - finished.start, browserCpu, status: finished.status }
}

/** Relative on/off differences of adjacent window pairs (in %), optionally without each window's first step. */
function windowPairs(windows, { trim = false } = {}) {
  const pairs = []
  for (let i = 0; i + 1 < windows.length; i += 2) {
    const [x, y] = [windows[i], windows[i + 1]]
    if (x.steps.length < (trim ? 2 : 1) || y.steps.length < (trim ? 2 : 1)) continue
    const on = (x.condition === "on" ? x : y).steps.slice(trim ? 1 : 0)
    const off = (x.condition === "on" ? y : x).steps.slice(trim ? 1 : 0)
    pairs.push((mean(on) / mean(off) - 1) * 100)
  }
  return pairs
}

async function measureInterleaved(browser) {
  await waitIdle()
  const options = { mute: args.mute ?? "/tmp/remi-perf-mute", windowMs: Number(args.window ?? 15) * 1000, steps: Number(args.steps ?? 160), batchCount: BATCH }
  const repeat = Math.max(1, Number(args.repeat ?? 1))
  const maxRepeat = Math.max(repeat, Number(args["max-repeat"] ?? repeat))
  const target = Number(args["target-upper95"] ?? 1)
  const jobs = []
  const from = Date.now()
  let decision
  for (let r = 0; ; r++) {
    if (r) await waitIdle()
    const run = await interleavedJob(browser, options)
    const pairs = windowPairs(run.windows)
    jobs.push({ ...run, pairs })
    const pooled = pairStats(jobs.flatMap((job) => job.pairs))
    decision = repeatDecision({ jobs: jobs.length, minJobs: repeat, maxJobs: maxRepeat, target, stats: pooled })
    log(`interleaved: job ${r + 1}/${maxRepeat > repeat ? `${repeat}-${maxRepeat}` : repeat}: ${run.status}, ${pairs.length} window pairs ${pairStats(pairs).mean} % (upper 95 % ${pairStats(pairs).upper95} %), ${run.painted} of ${run.received} previews painted; pooled ${pooled.n} pairs ${pooled.mean} %, upper 95 % ${pooled.upper95} %`)
    if (decision.stop) break
  }
  const traceUntil = Date.now() / 1000
  const windows = jobs.flatMap((job) => job.windows)
  const painted = jobs.reduce((n, job) => n + job.painted, 0)
  const received = jobs.reduce((n, job) => n + job.received, 0)
  const jobMs = jobs.reduce((n, job) => n + job.jobMs, 0)
  // the "on" condition is only valid if the page painted what it received (every image of a batch)
  const paintedShare = round(painted / Math.max(1, received), 3)
  if (paintedShare < 0.9) log(`interleaved: WARNING the page painted ${painted} of ${received} preview frames: the "on" windows did not show the previews`)
  // pairs never span two jobs: each job's windows are paired on their own
  const pairs = jobs.flatMap((job) => job.pairs)
  const trimmedPairs = jobs.flatMap((job) => windowPairs(job.windows, { trim: true }))
  const sd = Math.sqrt(pairs.reduce((n, v) => n + (v - mean(pairs)) ** 2, 0) / Math.max(1, pairs.length - 1))
  const ci = tQuantile(T95, 1.96, pairs.length - 1) * (sd / Math.sqrt(pairs.length))
  const usable = windows.filter((w) => w.steps.length)
  const by = (condition) => usable.filter((w) => w.condition === condition).flatMap((w) => w.steps)
  const on = by("on")
  const off = by("off")
  results.interleaved = {
    steps: options.steps,
    size: SIZE,
    windowS: options.windowMs / 1000,
    repeat,
    maxRepeat,
    targetUpper95: target,
    // whether the pooled pairs settle the budget (the bound of pairStats) and why the run stopped
    verdict: decision.verdict,
    stopReason: decision.reason,
    // the run's time range in the server trace (unix s): tools/perf/step_segments.py --run
    traceSince: from / 1000,
    traceUntil,
    previewClients: previewClients(from, "interleaved"),
    jobMs,
    painted,
    received,
    paintedShare,
    browserCpuPct: round((jobs.reduce((n, job) => n + job.browserCpu, 0) / (jobMs / 1000)) * 100, 1),
    stepMsOn: summary(on),
    stepMsOff: summary(off),
    overheadPct: round((mean(on) / mean(off) - 1) * 100, 2),
    batchCount: options.batchCount,
    pairs: pairs.map((v) => round(v, 2)),
    pairMeanPct: round(mean(pairs), 2),
    pairCi95Pct: round(ci, 2),
    pairStats: pairStats(pairs),
    trimmedPairStats: pairStats(trimmedPairs),
    medianOverheadPct: round((quantile(on, 0.5) / quantile(off, 0.5) - 1) * 100, 2),
    trimmedPairs: trimmedPairs.map((v) => round(v, 2)),
    // per job, its windows' index range in `windows` (tools/perf/pool_interleaved.py pools runs)
    jobs: jobs.map((job, index) => ({ status: job.status, jobMs: job.jobMs, painted: job.painted, received: job.received, windowCount: job.windows.length, firstWindow: jobs.slice(0, index).reduce((n, other) => n + other.windows.length, 0), pairStats: pairStats(job.pairs) })),
    windows: windows.map((w) => ({ condition: w.condition, steps: w.steps.map((v) => Math.round(v)) })),
  }
  if (decision.verdict === "inconclusive") {
    const needed = pairsNeeded({ sd, expected: Math.max(0, mean(pairs)), target })
    log(`interleaved: inconclusive after ${jobs.length} jobs: at this scatter (pair SD ${round(sd, 2)} %) about ${needed ?? "unbounded"} window pairs are needed (have ${pairs.length}); raise --max-repeat or measure on a quieter machine`)
  }
  log(`interleaved: on ${round(mean(on))} ms/step (n ${on.length}) vs off ${round(mean(off))} ms/step (n ${off.length}) -> ${results.interleaved.overheadPct} %; ${pairs.length} window pairs ${results.interleaved.pairMeanPct} ± ${results.interleaved.pairCi95Pct} % (95 % CI, one-sided upper bound ${results.interleaved.pairStats.upper95} %); first step of each window dropped: ${results.interleaved.trimmedPairStats.mean} ± ${results.interleaved.trimmedPairStats.ci95} % (upper ${results.interleaved.trimmedPairStats.upper95} %); medians ${results.interleaved.medianOverheadPct} %; ${painted} of ${received} previews painted; verdict ${decision.verdict} (${decision.reason})`)
  log(`interleaved: the server's cost per step (the primary figure): python tools/perf/step_segments.py ${TRACE} --run ${RESULT_FILE}`)
}

// ---------------------------------------------------------------------------- main

// --channel chrome: the installed Google Chrome instead of Playwright's Chromium (a headed run on
// the real display without downloading Chrome for Testing)
const browser = await chromium.launch({ headless: !args.headed, ...(args.channel ? { channel: args.channel } : {}), args: ["--disable-features=CalculateNativeWinOcclusion", "--disable-background-timer-throttling", "--disable-renderer-backgrounding"] })
try {
  results.browser = browser.version()
  for (const [name, run] of [
    ["load", measureLoad],
    ["inp", measureInp],
    ["networks", measureNetworks],
    ["gallery", measureGallery],
    ["preview", measurePreview],
    ["overhead", measureOverhead],
    ["interleaved", measureInterleaved],
  ]) {
    if (!ONLY.has(name)) continue
    try {
      await run(browser)
    } catch (error) {
      results[name] = { error: String(error?.stack ?? error) }
      log(`${name} failed:`, error)
    }
  }
} finally {
  await browser.close()
  const file = RESULT_FILE
  writeFileSync(file, JSON.stringify(results, null, 1))
  log(`results: ${file}`)
}
