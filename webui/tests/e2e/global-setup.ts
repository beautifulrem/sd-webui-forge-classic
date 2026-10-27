/**
 * Warms the Vite dev server before the workers start: Vite transforms modules on first request
 * (Babel with the React Compiler), and several workers opening their first page at once on a busy
 * machine could wait longer than an expectation's timeout for the same cold transforms. One
 * browser visits every page once so the workers start from cached transforms.
 */
import { chromium, type FullConfig } from "@playwright/test"

const ROUTES = ["txt2img", "img2img", "extras", "png-info", "gallery", "networks", "queue", "settings", "extensions", "ext/toolbox", "ext/json-builder"]

export default async function globalSetup(config: FullConfig) {
  const baseURL = config.projects[0]?.use.baseURL
  if (!baseURL) return
  const browser = await chromium.launch()
  try {
    const page = await browser.newPage()
    for (const route of ROUTES) {
      await page.goto(`${baseURL}/#/${route}`)
      const main = page.getByRole("main")
      await main.waitFor({ timeout: 120_000 })
      await page.waitForFunction(() => !document.querySelector("main [aria-busy=true]") && document.querySelector("main *"), undefined, { timeout: 120_000 })
    }
  } finally {
    await browser.close()
  }
}
