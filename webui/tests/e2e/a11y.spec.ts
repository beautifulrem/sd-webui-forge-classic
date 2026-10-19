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

    test("prompt highlighting keeps every token readable", async ({ page }) => {
      await openRoute(page, "txt2img")
      const prompt = page.getByRole("textbox", { name: "Prompt", exact: true })
      await prompt.click()
      await page.keyboard.press(`${process.platform === "darwin" ? "Meta" : "Control"}+A`)
      await page.keyboard.type("((a cat:1.2)), [day:night:0.5], <lora:watercolor_pastel_v1_0:0.8>, __colors__, BREAK, \\(paren\\), [[b]]")
      await page.locator("body").click({ position: { x: 5, y: 5 } }) // no selection highlight
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

    test("header and form popups are named dialogs without violations", async ({ page }) => {
      // Quick settings appear once a setting is pinned.
      await openRoute(page, "settings")
      await page.getByRole("button", { name: "Pin Automatically save every generated image to quick settings" }).click()
      await page.evaluate(() => (location.hash = "#/txt2img"))
      await expect(page.getByRole("textbox", { name: "Prompt", exact: true })).toBeVisible()
      const popups = [
        { trigger: page.getByRole("button", { name: "Quick settings" }), dialog: "Quick settings" },
        { trigger: page.getByRole("button", { name: "VAE / Text encoders" }), dialog: "VAE / Text encoders" },
        { trigger: page.getByRole("combobox", { name: "Checkpoint" }), dialog: "Checkpoint" },
        { trigger: page.getByRole("main").getByRole("combobox", { name: "Sampling method" }), dialog: "Sampling method" },
      ]
      for (const popup of popups) {
        await popup.trigger.click()
        await expect(page.getByRole("dialog", { name: popup.dialog })).toBeVisible()
        expect(await scan(page), popup.dialog).toEqual([])
        await page.keyboard.press("Escape")
        await expect(page.getByRole("dialog", { name: popup.dialog })).toBeHidden()
      }
    })
  })
}

/** Tabs through a page and lists the tab stops that show no focus indicator (outline or ring). */
async function stopsWithoutFocusRing(page: Page, presses: number): Promise<string[]> {
  const missing = new Set<string>()
  const check = () =>
    page.evaluate(() => {
      const element = document.activeElement as HTMLElement | null
      if (!element || element === document.body) return null
      const shows = (node: Element) => {
        const style = getComputedStyle(node)
        return (style.outlineStyle !== "none" && parseFloat(style.outlineWidth) > 0) || (style.boxShadow !== "none" && style.boxShadow !== "")
      }
      // a ring may be drawn by a wrapper that has the focus inside it (inputs in a group, the prompt editor)
      for (let node: Element | null = element, depth = 0; node && depth < 5; node = node.parentElement, depth++) {
        if (shows(node) && (node === element || node.matches(":focus-within"))) return null
      }
      const name = element.getAttribute("aria-label") || element.textContent?.trim().slice(0, 40) || element.tagName
      return `${element.tagName.toLowerCase()}[${element.getAttribute("role") ?? ""}] ${name}`
    })
  for (let i = 0; i < presses; i++) {
    await page.keyboard.press("Tab")
    let problem = await check()
    if (problem) {
      // some widgets mark focus a frame later (CodeMirror's .cm-focused): look again
      await page.waitForTimeout(150)
      problem = await check()
    }
    if (problem) missing.add(problem)
  }
  return [...missing]
}

test.describe("visible focus", () => {
  for (const route of ["txt2img", "extensions", "settings", "gallery", "queue", "png-info", "networks"]) {
    test(`every tab stop on #/${route} shows where the focus is`, async ({ page }) => {
      await openRoute(page, route)
      await page.waitForLoadState("networkidle")
      // rings fade in (box-shadow transitions): read their end state
      await page.addStyleTag({ content: "*, *::before, *::after { transition: none !important; animation: none !important; }" })
      expect(await stopsWithoutFocusRing(page, 70)).toEqual([])
    })
  }
})

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
