// Full-height screenshot of a page with every section expanded (inner scroll panes included):
//   node tools/audit/tall-shot.mjs <route> <out.png> [theme] [locale] [width] [height]
import { chromium } from "@playwright/test"
const [route, out, theme = "light", locale = "en", w = "1280", h = "9000"] = process.argv.slice(2)
const browser = await chromium.launch()
const context = await browser.newContext({ viewport: { width: +w, height: +h } })
await context.addInitScript(({ theme, locale }) => {
  localStorage.setItem("remi-appearance", JSON.stringify({ state: { theme, density: "comfortable", accent: "iris" }, version: 1 }))
  localStorage.setItem("remi-locale", JSON.stringify({ state: { chosen: locale }, version: 1 }))
}, { theme, locale })
const page = await context.newPage()
const errors = []
page.on("console", (m) => ["error", "warning"].includes(m.type()) && errors.push(m.text()))
await page.goto(`${process.env.BASE ?? "http://127.0.0.1:7860"}/#/${route}`)
await page.waitForLoadState("networkidle")
for (let round = 0; round < 4; round++) {
  const b = page.locator('main button[aria-expanded="false"]:not([aria-haspopup]):not([role="combobox"]):visible')
  const n = await b.count()
  if (!n) break
  for (let i = n - 1; i >= 0; i--) await b.nth(i).click({ timeout: 2000 }).catch(() => {})
  await page.waitForTimeout(400)
}
await page.waitForTimeout(800)
await page.screenshot({ path: out, fullPage: true })
console.log(errors.length ? errors.join("\n") : "no console errors")
await browser.close()
