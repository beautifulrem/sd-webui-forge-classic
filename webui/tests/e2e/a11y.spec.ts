/**
 * axe-core scans of every page in light and dark: no serious or critical
 * violations (WCAG 2.x A/AA rules plus axe best practices).
 */
import AxeBuilder from "@axe-core/playwright"
import { expect, type Page, test } from "@playwright/test"
import { openRoute, presetStorage, type Theme } from "./helpers"

/** A 96×64 solid PNG. */
const SOURCE_PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAGAAAABACAIAAABqVuVZAAAAaUlEQVR42u3QMQ0AAAgDsElDGtKQhgNujiZV0FQPhygQJEiQIEGCBAlCkCBBggQJEiQIQYIECRIkSJAgBAkSJEiQIEGCBCFIkCBBggQJEoQgQYIECRIkSBCCBAkSJEiQIEGCECRIkKB/Fn+AAh1bHed2AAAAAElFTkSuQmCC",
  "base64",
)

const PAGES = [
  "txt2img",
  "img2img",
  "extras",
  "png-info",
  "gallery",
  "networks",
  "queue",
  "settings",
  "extensions",
  "ext/toolbox",
  "no-such-page",
] as const

async function scan(page: Page) {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"])
    .analyze()
  return results.violations
    .filter((violation) => violation.impact === "serious" || violation.impact === "critical")
    .map((violation) => ({
      id: violation.id,
      impact: violation.impact,
      help: violation.help,
      nodes: violation.nodes.slice(0, 5).map((node) => `${node.target.join(" ")} :: ${node.failureSummary?.split("\n").slice(1, 3).join(" ")}`),
    }))
}

for (const theme of ["light", "dark"] satisfies Theme[]) {
  test.describe(`axe (${theme})`, () => {
    test.beforeEach(async ({ page }) => {
      await presetStorage(page, { theme })
    })

    for (const route of PAGES) {
      test(`#/${route} has no serious or critical violations`, async ({ page }) => {
        await openRoute(page, route)
        await expect(page.locator("html")).toHaveClass(theme === "dark" ? /\bdark\b/ : /^(?!.*\bdark\b)/)
        // Let lazy panels, fonts and query results settle before scanning.
        await page.waitForLoadState("networkidle")
        expect(await scan(page)).toEqual([])
      })
    }

    test("generate states: networks drawer, result, image viewer", async ({ page }) => {
      await openRoute(page, "txt2img")
      await page.locator("body").press("Alt+N")
      await expect(page.getByRole("dialog", { name: "Networks" }).getByRole("option").first()).toBeVisible()
      expect(await scan(page)).toEqual([])
      await page.getByRole("dialog", { name: "Networks" }).getByRole("button", { name: "Close" }).click()
      await page.getByRole("button", { name: /^Generate/ }).first().click()
      await expect(page.getByRole("img", { name: "Generated image" })).toBeVisible({ timeout: 20_000 })
      expect(await scan(page)).toEqual([])
      await page.getByRole("button", { name: "Open image viewer" }).first().click()
      await expect(page.getByRole("dialog", { name: "Image viewer" })).toBeVisible()
      expect(await scan(page)).toEqual([])
    })

    test("library states: gallery details, PNG Info result, queue with jobs", async ({ page }) => {
      await openRoute(page, "gallery")
      await page.getByRole("list", { name: "Images" }).getByRole("button").first().click()
      const details = page.getByRole("complementary", { name: "Image details" })
      await expect(details).toBeVisible()
      expect(await scan(page)).toEqual([])
      await details.getByRole("button", { name: "PNG Info" }).click()
      await expect(page.getByRole("tab", { name: /All metadata/ })).toBeVisible()
      expect(await scan(page)).toEqual([])
      await page.evaluate(() => (location.hash = "#/queue"))
      await page.getByRole("button", { name: "Pause" }).click()
      await page.evaluate(() => (location.hash = "#/txt2img"))
      await page.getByRole("button", { name: "Add to queue" }).click()
      await page.evaluate(() => (location.hash = "#/queue"))
      await expect(page.getByRole("list", { name: "Queued jobs" }).getByRole("listitem")).toHaveCount(1)
      expect(await scan(page)).toEqual([])
    })

    test("img2img editor in inpaint mode", async ({ page }) => {
      await openRoute(page, "img2img")
      const chooser = page.waitForEvent("filechooser")
      await page.getByRole("button", { name: "Add a source image" }).click()
      await (await chooser).setFiles({ name: "source.png", mimeType: "image/png", buffer: SOURCE_PNG })
      await expect(page.getByRole("img", { name: /^Image editor, / })).toBeVisible()
      await page.getByRole("radiogroup", { name: "Mode" }).getByRole("radio", { name: "Inpaint", exact: true }).click()
      await expect(page.getByRole("toolbar", { name: "Tools" })).toBeVisible()
      expect(await scan(page)).toEqual([])
    })

    test("settings with unsaved changes", async ({ page }) => {
      await openRoute(page, "settings")
      await page.getByRole("switch", { name: "Automatically save every generated image", exact: true }).click()
      await expect(page.getByRole("region", { name: "Unsaved changes" })).toBeVisible()
      expect(await scan(page)).toEqual([])
    })

    test("open overlays have no serious or critical violations", async ({ page }) => {
      await openRoute(page, "txt2img")
      await page.getByRole("button", { name: "Appearance" }).click()
      await expect(page.getByRole("menu")).toBeVisible()
      expect(await scan(page)).toEqual([])
      await page.keyboard.press("Escape")
      await page.keyboard.press(process.platform === "darwin" ? "Meta+K" : "Control+K")
      await expect(page.getByRole("dialog")).toBeVisible()
      expect(await scan(page)).toEqual([])
    })
  })
}

test.describe("axe on a phone", () => {
  for (const route of ["txt2img", "img2img", "gallery", "queue", "settings", "networks"]) {
    test(`#/${route} has no serious or critical violations @phone`, async ({ page }) => {
      await presetStorage(page, { theme: "dark" })
      await openRoute(page, route)
      await page.waitForLoadState("networkidle")
      expect(await scan(page)).toEqual([])
    })
  }
})
