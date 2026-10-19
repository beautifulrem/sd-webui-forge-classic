/**
 * Layout and feedback checks that need a real browser: the header at
 * 1280×720 (idle and while a job runs), tab strips on a phone, platform
 * shortcut hints and generation warnings.
 */
import { expect, type Page, test } from "@playwright/test"
import { mod, openGenerate, presetStorage, setPrompt } from "./helpers"

/** Pairs of header controls whose boxes overlap, and kbd hints cut by their button. */
async function headerProblems(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const header = document.querySelector("header")!
    const boxes = [...header.querySelectorAll<HTMLElement>("button, [role=combobox], h1")]
      .map((element) => ({ element, rect: element.getBoundingClientRect(), name: (element.getAttribute("aria-label") || element.textContent || "").trim() }))
      .filter(({ rect }) => rect.width > 0)
    const problems: string[] = []
    for (const [i, a] of boxes.entries()) {
      for (const b of boxes.slice(i + 1)) {
        if (a.element.contains(b.element) || b.element.contains(a.element)) continue
        const x = Math.min(a.rect.right, b.rect.right) - Math.max(a.rect.left, b.rect.left)
        const y = Math.min(a.rect.bottom, b.rect.bottom) - Math.max(a.rect.top, b.rect.top)
        if (x > 0.5 && y > 0.5) problems.push(`"${a.name}" overlaps "${b.name}" by ${x.toFixed(1)}px`)
      }
      if (a.rect.right > innerWidth + 0.5) problems.push(`"${a.name}" leaves the window`)
    }
    for (const kbd of document.querySelectorAll<HTMLElement>("header button kbd, main button kbd")) {
      const k = kbd.getBoundingClientRect()
      const b = kbd.closest("button")!.getBoundingClientRect()
      if (k.width > 0 && (k.right > b.right - 2 || k.left < b.left)) problems.push(`kbd "${kbd.textContent}" is cut by its button`)
    }
    for (const label of document.querySelectorAll<HTMLElement>("main button .truncate")) {
      if (label.scrollWidth > label.clientWidth + 1) problems.push(`button label "${label.textContent}" is cut`)
    }
    return problems
  })
}

test.describe("layout at 1280×720", () => {
  test.use({ viewport: { width: 1280, height: 720 } })

  for (const locale of ["en", "zh-CN"]) {
    test(`the header and the generate bar fit, idle and while a job runs (${locale})`, async ({ page }) => {
      await presetStorage(page, { locale })
      // Quick settings in the header too (the most crowded case).
      await page.goto("/#/settings")
      await page.getByRole("button", { name: /^(Pin .*quick settings|将.*固定)/ }).first().click()
      await expect(page.getByRole("button", { name: /^(Unpin .*quick settings|从快捷设置中取消固定)/ }).first()).toBeVisible()
      await page.evaluate(() => (location.hash = "#/txt2img"))
      await expect(page.getByRole("textbox", { name: /^(Prompt|提示词)$/ })).toBeVisible()
      await expect(page.locator("header").getByRole("button", { name: /^(Quick settings|快捷设置)$/ })).toBeVisible()
      await expect.poll(() => headerProblems(page)).toEqual([])

      await page.getByRole("button", { name: /^(Generate|生成)/ }).first().click()
      await expect(page.getByRole("button", { name: /^(Interrupt|中断)$/ })).toBeVisible()
      expect(await headerProblems(page)).toEqual([])
    })
  }
})

test.describe("tab strips on a phone", () => {
  test.use({ viewport: { width: 420, height: 860 } })

  for (const [route, label] of [
    ["extensions", "Extensions"],
    ["networks", "Network kinds"],
  ] as const) {
    test(`#/${route} shows its tabs as a select instead of cutting them`, async ({ page }) => {
      await page.goto(`/#/${route}`)
      const select = page.getByRole("main").getByRole("combobox", { name: label })
      await expect(select).toBeVisible()
      const overflowing = await page.evaluate(() =>
        [...document.querySelectorAll<HTMLElement>("[role=tablist]")]
          .filter((list) => list.getBoundingClientRect().width > 0 && getComputedStyle(list).visibility !== "hidden")
          .filter((list) => list.scrollWidth > list.clientWidth + 1)
          .map((list) => list.getAttribute("aria-label")),
      )
      expect(overflowing).toEqual([])
      // The select switches tabs like the strip.
      await select.click()
      const second = page.locator("[data-slot=select-item]").nth(1)
      const name = (await second.textContent())!.trim()
      await second.click()
      await expect(select).toHaveText(new RegExp(name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")))
    })
  }

  test("the tabs come back when there is room", async ({ page }) => {
    await page.goto("/#/extensions")
    await expect(page.getByRole("main").getByRole("combobox", { name: "Extensions" })).toBeVisible()
    await page.setViewportSize({ width: 1280, height: 860 })
    await expect(page.getByRole("tab", { name: "Backup and restore" })).toBeVisible()
    await expect(page.getByRole("main").getByRole("combobox", { name: "Extensions" })).toHaveCount(0)
  })
})

test.describe("hints and warnings", () => {
  test("the prompt placeholder names this platform's shortcuts", async ({ page }) => {
    await openGenerate(page)
    const prompt = await setPrompt(page, "x")
    await prompt.press("Backspace") // the placeholder shows in an empty prompt
    const generate = process.platform === "darwin" ? "⌘Enter" : "Ctrl+Enter"
    await expect(page.locator(".cm-placeholder").first()).toContainText(`${generate} to Generate`)
    await expect(page.getByRole("button", { name: /^Generate/ }).first()).toContainText(generate)
  })

  test("a LoRA that is not there ends the job with a warning", async ({ page }) => {
    await openGenerate(page)
    await setPrompt(page, "1girl, silver hair, <lora:doesnotexist:1>")
    await page.keyboard.press(`${mod}+Enter`)
    const toast = page.locator("[data-sonner-toast]").filter({ hasText: "Finished with warnings" })
    await expect(toast).toBeVisible({ timeout: 20_000 })
    await expect(toast).toContainText("LoRA not found or failed to load: doesnotexist")
  })
})

test.describe("schema rows on extension pages", () => {
  const actions = ["Build prompt", "Send to txt2img Prompt", "Replace txt2img Prompt", "Clear all fields"]

  test("an action row is one left-aligned group of same-sized buttons; status lines use the app's status style", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 900 })
    await page.goto("/#/ext/toolbox")
    const main = page.getByRole("main")
    const boxes = []
    for (const name of actions) boxes.push((await main.getByRole("button", { name, exact: true }).boundingBox())!)
    expect(new Set(boxes.map((box) => Math.round(box.height))).size).toBe(1)
    for (const [index, box] of boxes.slice(1).entries()) expect(box.x - (boxes[index]!.x + boxes[index]!.width)).toBeLessThan(16) // side by side, not spread out
    await main.getByRole("button", { name: "Build prompt" }).click()
    const status = main.locator("[data-tone=success]")
    await expect(status).toHaveText("Prompt built") // the "✅" became the icon and tone
  })

  test("two columns of fields stack on a phone", async ({ page }) => {
    for (const [width, sideBySide] of [
      [1280, true],
      [420, false],
    ] as const) {
      await page.setViewportSize({ width, height: 900 })
      await page.goto("/#/ext/toolbox")
      const quality = (await page.getByRole("textbox", { name: "Quality" }).boundingBox())!
      const series = (await page.getByRole("textbox", { name: "Series" }).boundingBox())!
      expect(Math.abs(quality.y - series.y) < 2, `${width}px`).toBe(sideBySide)
    }
  })
})

