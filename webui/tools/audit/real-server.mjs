// Browser audit of every page against a running WebUI (docs/remi-webui/integration-checklist.md).
//
//   node tools/audit/real-server.mjs --base http://127.0.0.1:7860 --out /tmp/remi-int/audit \
//     [--routes txt2img,gallery] [--sizes 1280x720,1920x1080,420x860] [--themes light,dark]
//     [--locales en,zh-CN] [--axe all|first|none] [--expand 1] [--headed]
//
// For each route × viewport × theme × locale it loads the page, waits for it to settle and records:
// console errors / warnings, uncaught page errors, failed requests (status ≥ 400), axe serious /
// critical violations, page-level horizontal overflow, wrapped horizontal tab strips, clipped text
// (overflowing an overflow:hidden box without an ellipsis), and in zh-CN visible text with Latin
// words and no CJK (to triage mixed-language labels). With --expand, every collapsed section in
// the main landmark is opened first (generate pages: every built-in extension panel). A full-page
// screenshot is written per combination; results go to <out>/report.json, exit 1 on any finding
// other than the zh-CN Latin list (which needs a human to separate technical names).
import { mkdirSync, writeFileSync } from "node:fs"
import path from "node:path"
import AxeBuilder from "@axe-core/playwright"
import { chromium } from "@playwright/test"

const args = Object.fromEntries(
  process.argv.slice(2).reduce((pairs, arg, i, all) => {
    if (arg.startsWith("--")) pairs.push([arg.slice(2), all[i + 1] && !all[i + 1].startsWith("--") ? all[i + 1] : "1"])
    return pairs
  }, []),
)
const BASE = args.base ?? "http://127.0.0.1:7860"
const OUT = args.out ?? "/tmp/remi-int/audit"
const SIZES = (args.sizes ?? "1280x720,1920x1080,420x860").split(",").map((s) => s.split("x").map(Number))
const THEMES = (args.themes ?? "light,dark").split(",")
const LOCALES = (args.locales ?? "en,zh-CN").split(",")
const AXE = args.axe ?? "first"
const EXPAND = args.expand === "1"
mkdirSync(path.join(OUT, "shots"), { recursive: true })

const app = await (await fetch(`${BASE}/api/v2/app`)).json()
const ROUTES = args.routes
  ? args.routes.split(",")
  : ["txt2img", "img2img", "extras", "png-info", "gallery", "networks", "queue", "settings", "extensions", ...app.pages.map((p) => `ext/${p.id}`), "no-such-page"]

/** Runs in the page: layout and language findings. */
function inspect(locale) {
  const vw = document.documentElement.clientWidth
  const visible = (el) => {
    const r = el.getBoundingClientRect()
    if (r.width === 0 || r.height === 0) return false
    const s = getComputedStyle(el)
    return s.visibility !== "hidden" && s.display !== "none" && Number(s.opacity) > 0
  }
  const describe = (el) => {
    const id = el.id ? `#${el.id}` : ""
    const label = el.getAttribute("aria-label") ?? el.textContent?.trim().slice(0, 40) ?? ""
    return `${el.tagName.toLowerCase()}${id} "${label}"`
  }
  const out = { overflowX: document.documentElement.scrollWidth - vw, wrappedTabs: [], clipped: [], latin: [] }
  for (const list of document.querySelectorAll('[role="tablist"]')) {
    if (!visible(list) || list.getAttribute("aria-orientation") === "vertical") continue
    const tops = new Set([...list.querySelectorAll('[role="tab"]')].filter(visible).map((t) => Math.round(t.getBoundingClientRect().top)))
    if (tops.size > 1) out.wrappedTabs.push(describe(list))
  }
  for (const el of document.querySelectorAll("button, label, a, h1, h2, h3, h4, [role=tab], [role=option], span, p, div")) {
    if (el.children.length > 0 && el.tagName === "DIV") continue
    if (!visible(el) || !el.textContent?.trim()) continue
    const s = getComputedStyle(el)
    if (s.textOverflow === "ellipsis" || el.closest("[data-allow-clip], .cm-editor")) continue
    const r = el.getBoundingClientRect()
    if (r.width <= 2 || r.height <= 2 || (s.clip && s.clip !== "auto") || s.clipPath !== "none") continue // screen-reader-only text
    const clipsX = s.overflowX === "hidden" || s.overflowX === "clip"
    const clipsY = s.overflowY === "hidden" || s.overflowY === "clip"
    if ((clipsX && el.scrollWidth > el.clientWidth + 2) || (clipsY && el.scrollHeight > el.clientHeight + 2 && !s.webkitLineClamp?.match(/\d/))) {
      if (el.clientWidth > 0 && el.clientHeight > 0) out.clipped.push(describe(el))
    }
  }
  if (locale !== "en") {
    const walker = document.createTreeWalker(document.querySelector("#root") ?? document.body, NodeFilter.SHOW_TEXT)
    const seen = new Set()
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const text = node.textContent.trim()
      const parent = node.parentElement
      if (!text || !parent || !visible(parent)) continue
      if (parent.closest('textarea, input, code, pre, kbd, .cm-editor, [translate="no"], [data-raw-text]')) continue
      if (/[㐀-鿿]/.test(text) || !/[A-Za-z]{3,}/.test(text)) continue
      if (!seen.has(text)) seen.add(text)
    }
    out.latin = [...seen]
  }
  return out
}

/** Opens every collapsed disclosure in the main landmark (repeated: opening one can reveal more). */
async function expandAll(page) {
  for (let round = 0; round < 4; round++) {
    const buttons = page.locator('main button[aria-expanded="false"]:not([aria-haspopup]):not([role="combobox"]):visible')
    const count = await buttons.count()
    if (!count) return
    for (let i = count - 1; i >= 0; i--) await buttons.nth(i).click({ timeout: 2000 }).catch(() => {})
    await page.waitForTimeout(400)
  }
}

const browser = await chromium.launch({ headless: args.headed !== "1" })
const report = []
let failures = 0
for (const locale of LOCALES) {
  for (const theme of THEMES) {
    for (const [width, height] of SIZES) {
      const context = await browser.newContext({ viewport: { width, height }, locale: locale === "en" ? "en-US" : locale })
      await context.addInitScript(
        ({ theme, locale }) => {
          localStorage.setItem("remi-appearance", JSON.stringify({ state: { theme, density: "comfortable", accent: "iris" }, version: 1 }))
          localStorage.setItem("remi-locale", JSON.stringify({ state: { chosen: locale }, version: 1 }))
        },
        { theme, locale },
      )
      const page = await context.newPage()
      let messages = []
      page.on("console", (m) => {
        if (m.type() === "error" || m.type() === "warning") messages.push(`${m.type()}: ${m.text()}`)
      })
      page.on("pageerror", (e) => messages.push(`pageerror: ${e.message}`))
      page.on("response", (r) => {
        if (r.status() >= 400 && !r.url().includes("/no-such")) messages.push(`http ${r.status()}: ${r.url()}`)
      })
      for (const route of ROUTES) {
        messages = []
        await page.goto(`${BASE}/#/${route}`)
        await page.waitForLoadState("networkidle").catch(() => {})
        await page.locator("main [aria-busy=true]").first().waitFor({ state: "detached", timeout: 10_000 }).catch(() => {})
        await page.waitForTimeout(600)
        if (EXPAND) await expandAll(page)
        const layout = await page.evaluate(inspect, locale)
        const runAxe = AXE === "all" || (AXE === "first" && width === SIZES[0][0])
        let axe = []
        if (runAxe) {
          const result = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"]).analyze()
          axe = result.violations
            .filter((v) => v.impact === "serious" || v.impact === "critical")
            .map((v) => `${v.id}: ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`)
        }
        const name = `${route.replace(/\//g, "_")}-${width}x${height}-${theme}-${locale}${EXPAND ? "-expanded" : ""}`
        await page.screenshot({ path: path.join(OUT, "shots", `${name}.png`), fullPage: true })
        const entry = { route, width, height, theme, locale, console: messages, axe, ...layout }
        const bad = messages.length + axe.length + layout.wrappedTabs.length + layout.clipped.length + (layout.overflowX > 0 ? 1 : 0)
        if (bad) failures++
        report.push(entry)
        console.log(`${bad ? "FAIL" : "ok  "} ${name}${bad ? ` ${JSON.stringify({ console: messages, axe, overflowX: layout.overflowX, wrappedTabs: layout.wrappedTabs, clipped: layout.clipped.slice(0, 6) })}` : ""}${layout.latin.length ? ` latin=${layout.latin.length}` : ""}`)
      }
      await context.close()
    }
  }
}
await browser.close()
writeFileSync(path.join(OUT, "report.json"), JSON.stringify(report, null, 1))
const latin = [...new Set(report.flatMap((r) => r.latin))]
writeFileSync(path.join(OUT, "latin.txt"), latin.join("\n"))
console.log(`${report.length} views, ${failures} with findings; ${latin.length} distinct Latin-only texts in non-English views (${path.join(OUT, "latin.txt")})`)
process.exit(failures ? 1 : 0)
