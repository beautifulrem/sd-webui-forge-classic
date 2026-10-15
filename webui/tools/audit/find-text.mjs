// Where a visible text appears on a page (all sections open): node path and nearest label.
//   node tools/audit/find-text.mjs <route> <locale> <text>
import { chromium } from "@playwright/test"
const [route, locale, text] = process.argv.slice(2)
const browser = await chromium.launch()
const context = await browser.newContext({ viewport: { width: 1280, height: 720 } })
await context.addInitScript((locale) => localStorage.setItem("remi-locale", JSON.stringify({ state: { chosen: locale }, version: 1 })), locale)
const page = await context.newPage()
await page.goto(`${process.env.BASE ?? "http://127.0.0.1:7860"}/#/${route}`)
await page.waitForLoadState("networkidle")
for (let round = 0; round < 4; round++) {
  const b = page.locator('main button[aria-expanded="false"]:not([aria-haspopup]):not([role="combobox"]):visible')
  const n = await b.count()
  if (!n) break
  for (let i = n - 1; i >= 0; i--) await b.nth(i).click({ timeout: 2000 }).catch(() => {})
  await page.waitForTimeout(400)
}
const found = await page.evaluate((text) => {
  const out = []
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT)
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (node.textContent.trim() !== text) continue
    const el = node.parentElement
    const field = el.closest("[data-slot], .group\\/field, section")
    const label = field?.querySelector("label, h2, h3, [id$='-title']")?.textContent?.trim()
    out.push(`${el.tagName}.${el.className.toString().slice(0, 50)} | label: ${label} | section: ${el.closest("section")?.querySelector("[id$='-title']")?.textContent}`)
  }
  return out
}, text)
console.log(found.join("\n") || "not found")
await browser.close()
