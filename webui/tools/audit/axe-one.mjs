import AxeBuilder from "@axe-core/playwright"
import { chromium } from "@playwright/test"
const [route, theme = "dark", locale = "en", w = "1280", h = "720"] = process.argv.slice(2)
const browser = await chromium.launch()
const context = await browser.newContext({ viewport: { width: +w, height: +h } })
await context.addInitScript(({ theme, locale }) => {
  localStorage.setItem("remi-appearance", JSON.stringify({ state: { theme, density: "comfortable", accent: "iris" }, version: 1 }))
  localStorage.setItem("remi-locale", JSON.stringify({ state: { chosen: locale }, version: 1 }))
}, { theme, locale })
const page = await context.newPage()
await page.goto(`http://127.0.0.1:7860/#/${route}`)
await page.waitForLoadState("networkidle"); await page.waitForTimeout(1000)
if (process.env.EXPAND) {
  for (let round = 0; round < 4; round++) {
    const b = page.locator('main button[aria-expanded="false"]:not([aria-haspopup]):not([role="combobox"]):visible'); const n = await b.count(); if (!n) break
    for (let i = n - 1; i >= 0; i--) await b.nth(i).click({ timeout: 2000 }).catch(() => {})
    await page.waitForTimeout(400)
  }
}
const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"]).analyze()
for (const v of r.violations.filter((v) => ["serious", "critical"].includes(v.impact))) {
  console.log(v.id, v.impact, v.help)
  for (const n of v.nodes.slice(0, 6)) console.log("  ", n.html.slice(0, 300), "\n    ", n.failureSummary.replace(/\n/g, " ").slice(0, 300))
}
await browser.close()
